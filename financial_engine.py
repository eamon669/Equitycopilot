"""解析、财务指标、可解释筛查、同业估值。纯函数不依赖 Streamlit。"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from io import BytesIO
import math
import re
import numpy as np
import pandas as pd
from mock_data import LABELS, STATEMENTS

MAX_BYTES = 20 * 1024 * 1024
UNITS = {'million':1.0,'百万元':1.0,'元':1e-6,'yuan':1e-6,'万元':.01,'亿元':100.0}
ALIASES = {v:k for k,v in LABELS.items()}
ALIASES.update({k:k for k in LABELS})
ALIASES.update({'归属于母公司股东的净利润':'parent_net_profit','归属于母公司所有者权益合计':'parent_equity',
    '经营活动产生的现金流量净额':'cfo','投资活动产生的现金流量净额':'cfi',
    '筹资活动产生的现金流量净额':'cff','汇率变动对现金及现金等价物的影响':'fx_effect',
    '期初现金及现金等价物余额':'cash_begin','期末现金及现金等价物余额':'cash_end'})
META = ['company','year','period','currency','unit','scope']


@dataclass
class Dataset:
    financials: pd.DataFrame
    provenance: list[dict]
    documents: list[dict]
    warnings: list[str]


@dataclass(frozen=True)
class Thresholds:
    cfo_high: float = .5
    cfo_medium: float = .8
    ar_high: float = .20
    ar_medium: float = .10
    dio_high: float = 30.0
    dio_medium: float = 15.0
    gm_change: float = .03
    balance_tolerance: float = .001

    def validate(self):
        if not (0 <= self.cfo_high < self.cfo_medium <= 2):
            raise ValueError('现金转化率阈值需满足 0 ≤ 高风险 < 中风险 ≤ 2。')
        if not (0 <= self.ar_medium < self.ar_high <= 2):
            raise ValueError('应收增速差阈值必须递增。')
        if not (0 <= self.dio_medium < self.dio_high):
            raise ValueError('周转天数阈值必须递增。')


def number(value) -> float:
    if value is None or pd.isna(value): return float('nan')
    s = str(value).strip().replace(',','').replace('，','').replace('−','-')
    if s in ('','-','--','—','N/A','NA','None'): return float('nan')
    if s.startswith('(') and s.endswith(')'): s = '-'+s[1:-1]
    try:
        result = float(s)
    except (ValueError,TypeError) as exc:
        raise ValueError(f'不能识别的数字：{str(value)[:50]}') from exc
    if not math.isfinite(result): raise ValueError('不接受无穷大或非有限数字。')
    return result


def ratio(numerator, denominator, positive: bool = True) -> float:
    if not (pd.notna(numerator) and pd.notna(denominator)): return np.nan
    if (denominator <= 0 if positive else denominator == 0): return np.nan
    value = numerator / denominator
    return float(value) if np.isfinite(value) else np.nan


def validate_financials(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: raise ValueError('没有可用财务数据，请导入标准 CSV。')
    required = set(META)
    if not required.issubset(df.columns): raise ValueError('缺少字段：'+', '.join(sorted(required-set(df.columns))))
    data = df.copy()
    if data['company'].nunique(dropna=False) != 1 or data['company'].isna().any():
        raise ValueError('每次分析仅允许一个明确的公司主体。')
    for key, expected in [('currency','CNY'),('period','FY'),('scope','consolidated'),('unit','million')]:
        if not data[key].eq(expected).all(): raise ValueError(f'{key} 必须统一为 {expected}；不自动混合口径。')
    years = pd.to_numeric(data['year'],errors='coerce')
    if years.isna().any() or (years % 1 != 0).any() or not years.between(1900,2100).all():
        raise ValueError('year 必须为 1900—2100 之间的整数。')
    data['year'] = years.astype(int)
    if data['year'].duplicated().any(): raise ValueError('同一年度重复，需人工确认重述/合并口径。')
    for key in LABELS:
        if key not in data: data[key] = np.nan
        else: data[key] = data[key].map(number)
    return data.sort_values('year').reset_index(drop=True)


def parse_csv(content: bytes, filename: str) -> Dataset:
    """严格长表模板。按原始物理行/Item 追溯；未知行警告，缺失项不补零。"""
    if len(content)>MAX_BYTES: raise ValueError('文件超过 20 MB 上限。')
    try: raw = pd.read_csv(BytesIO(content),encoding='utf-8-sig',dtype=str,keep_default_na=False)
    except UnicodeDecodeError:
        try: raw = pd.read_csv(BytesIO(content),encoding='gb18030',dtype=str,keep_default_na=False)
        except Exception as exc: raise ValueError('CSV 编码错误，请保存为 UTF-8。') from exc
    except Exception as exc: raise ValueError('CSV 文件损坏或格式错误。') from exc
    required = set(META+['statement','item','value'])
    if not required.issubset(raw): raise ValueError('请使用样例 CSV，缺少：'+', '.join(sorted(required-set(raw))))
    if len(raw)>20000: raise ValueError('最多支持 20,000 条财务记录。')
    warnings, prov, records, keys = [], [], [], set()
    for idx,row in raw.iterrows():
        item = ALIASES.get(row['item'].strip())
        if item is None:
            warnings.append(f'Row {idx+2}：忽略未知项目 {row["item"][:60]}'); continue
        if row['unit'] not in UNITS: raise ValueError(f'Row {idx+2}：不支持单位 {row["unit"]}')
        if row['statement'] != STATEMENTS[item]: raise ValueError(f'Row {idx+2}：{item} 三表分类错误。')
        year_float = number(row['year'])
        if pd.isna(year_float) or not year_float.is_integer(): raise ValueError('年度须为整数。')
        year = int(year_float)
        key = (row['company'],year,item)
        if key in keys: raise ValueError(f'重复项目 {year}/{item}，不能静默求和或覆盖。')
        keys.add(key)
        value = number(row['value'])*UNITS[row['unit']]
        records.append({**{k:row[k] for k in META},'year':year,'unit':'million','item':item,'value':value})
        prov.append(dict(year=year,item=item,value=value,source=filename,page=None,
            locator=f'Row {idx+2} / Item {item}',raw_value=row['value'],raw_unit=row['unit']))
    if not records: raise ValueError('CSV 未包含已识别财务项目。')
    long = pd.DataFrame(records)
    # 不使用 pivot_table，避免重复行聚合及全缺失列被静默删除。
    wide = long.set_index(META+['item'])['value'].unstack('item').reset_index()
    df = validate_financials(wide)
    if df[list(LABELS)].isna().any().any(): warnings.append('存在缺失财务科目；依赖它们的指标显示 N/A / 未评估。')
    return Dataset(df,prov,[],warnings)


def parse_pdf(content: bytes, filename: str, company: str) -> Dataset:
    """文本型 PDF 按页检索；财务表仅识别明确年度列与单位，拒绝猜测。扫描件需先 OCR。"""
    import pdfplumber
    if len(content)>MAX_BYTES: raise ValueError('文件超过 20 MB 上限。')
    if not content.startswith(b'%PDF'): raise ValueError('文件不是有效 PDF。')
    docs, candidates, warnings = [], [], []
    try:
        with pdfplumber.open(BytesIO(content)) as pdf:
            if len(pdf.pages)>200: raise ValueError('单份 PDF 最多 200 页，请先拆分。')
            for page_no,page in enumerate(pdf.pages,1):
                text = page.extract_text() or ''
                if not text.strip(): warnings.append(f'Page {page_no} 无文本层，未执行 OCR。'); continue
                for start in range(0,len(text),1000):
                    docs.append(dict(id=f'P{page_no}C{start//1000+1}',source=filename,page=page_no,
                                     item=f'段落 {start//1000+1}',text=text[start:start+1200]))
                unit_hits = set(re.findall(r'单位\s*[:：]\s*(?:人民币)?\s*(百万元|万元|亿元|元)',text))
                # 页内明确有不同币种或多种单位时，只保留文本，留待人工导入核实值。
                if len(unit_hits)!=1 or re.search(r'美元|港元|欧元|USD|HKD|EUR',text): continue
                unit = unit_hits.pop()
                if not re.search(r'人民币|CNY',text): continue
                if not re.search(r'合并',text): continue
                for table in page.extract_tables() or []:
                    header_idx, years = None, {}
                    for n,cells in enumerate(table[:4]):
                        found = {}
                        for col,cell in enumerate(cells):
                            s = str(cell or '').replace('\n','')
                            m = re.fullmatch(r'\s*((?:19|20)\d{2})(?:年度?|年12月31日|[/-]12[/-]31)?\s*',s)
                            if m: found[col]=int(m.group(1))
                        if found: header_idx,years=n,found; break
                    if header_idx is None: continue
                    for cells in table[header_idx+1:]:
                        label = re.sub(r'\s+','',str(cells[0] or ''))
                        item = ALIASES.get(label)
                        if item is None: continue
                        for col,year in years.items():
                            if col>=len(cells): continue
                            try: value=number(cells[col])*UNITS[unit]
                            except ValueError:
                                warnings.append(f'Page {page_no}/{item} 数字无法识别，已留空。');continue
                            candidates.append(dict(year=year,item=item,value=value,source=filename,
                                page=page_no,locator=f'Page {page_no} / Item {label}',
                                raw_value=str(cells[col]),raw_unit=unit))
    except ValueError: raise
    except Exception as exc: raise ValueError('PDF 解析失败，可能已加密、损坏或表格不受支持。') from exc
    if not docs: raise ValueError('未提取到文本。扫描 PDF 请先 OCR 后上传。')
    seen, ambiguous = {}, set()
    for c in candidates:
        key=(c['year'],c['item'])
        if key in seen: ambiguous.add(key)  # 重复/重述无法推断哪个是正确合并报表。
        else: seen[key]=c
    prov=[v for k,v in seen.items() if k not in ambiguous]
    if ambiguous: warnings.append('重复或重述科目已排除，请从 CSV 核实导入：'+str(sorted(ambiguous)))
    if prov:
        frame=pd.DataFrame(prov).pivot(index='year',columns='item',values='value').reset_index()
        for k,v in dict(company=company,period='FY',currency='CNY',unit='million',scope='consolidated').items(): frame[k]=v
        frame=validate_financials(frame)
        warnings.append('PDF 财务抽取为候选值，须逐项对照页码复核；确认后才可进入分析。')
    else:
        frame=pd.DataFrame()
        warnings.append('已提取文档文本；未识别可靠财务表，请同时导入标准 CSV。')
    return Dataset(frame,prov,docs,warnings)


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = validate_financials(df)
    prev = d.shift(1); consecutive = d.year.sub(prev.year).eq(1)
    d['gross_margin']=[1-ratio(c,r) for c,r in zip(d.cogs,d.revenue)]
    d['cfo_conversion']=[ratio(c,n) for c,n in zip(d.cfo,d.net_profit)]
    d['profit_cash_gap']=[ratio(n-c,abs(n)) for n,c in zip(d.net_profit,d.cfo)]
    for item,col in [('revenue','revenue_growth'),('receivables','ar_growth')]:
        d[col]=[ratio(v-p,p) if ok else np.nan for v,p,ok in zip(d[item],prev[item],consecutive)]
    d['ar_growth_gap']=d.ar_growth-d.revenue_growth
    d['dio']=[ratio((a+b)/2,c)*365 if ok else np.nan for a,b,c,ok in zip(d.inventory,prev.inventory,d.cogs,consecutive)]
    d['dio_change']=d.dio.diff().where(consecutive)
    d['gm_change']=d.gross_margin.diff().where(consecutive)
    d['roe']=[ratio(n,(a+b)/2) if ok else np.nan for n,a,b,ok in zip(d.parent_net_profit,d.parent_equity,prev.parent_equity,consecutive)]
    d['balance_gap']=d.assets-d.liabilities-d.equity
    d['cash_bridge_gap']=d.cash_end-d.cash_begin-d.cfo-d.cfi-d.cff-d.fx_effect
    d['cash_continuity_gap']=(d.cash_begin-prev.cash_end).where(consecutive)
    return d


def audit_financials(df: pd.DataFrame, provenance: list[dict], thresholds: Thresholds | None = None) -> list[dict]:
    t=thresholds or Thresholds();t.validate()
    m=calculate_metrics(df);r=m.iloc[-1];year=int(r.year)
    results=[]
    def add(code,title,level,value,formula,reason,items,years=None,threshold=''):
        relevant=years or [year]
        evidence=[p for p in provenance if p['item'] in items and p['year'] in relevant]
        inputs=[dict(year=int(row.year),item=k,value=None if pd.isna(row[k]) else float(row[k]))
                for _,row in m[m.year.isin(relevant)].iterrows() for k in items]
        results.append(dict(code=code,title=title,level=level,value=None if pd.isna(value) else float(value),
            formula=formula,reason=reason,threshold=threshold,inputs=inputs,sources=evidence))
    x=r.cfo_conversion
    level='未评估' if pd.isna(x) else '高' if x<t.cfo_high else '中' if x<t.cfo_medium else '低'
    add('cash_quality','利润现金转化',level,x,'CFO / 净利润；背离度 = (净利润 − CFO) / |净利润|',
        '净利润为非正或数据缺失时，比率不适用。现金流背离是核查线索，不能直接认定利润造假。',
        ['cfo','net_profit'],threshold=f'高 < {t.cfo_high:.0%}；中 < {t.cfo_medium:.0%}')
    x=r.ar_growth_gap
    add('receivables','应收与收入增速差','未评估' if pd.isna(x) else '高' if x>t.ar_high else '中' if x>t.ar_medium else '低',x,
        '应收账款同比增速 − 营业收入同比增速',
        '增速差偏高需结合信用政策、收入截止测试与期后回款核实。上年基数≤0时不计算。',
        ['receivables','revenue'],[year-1,year],f'高 > {t.ar_high:.0%}；中 > {t.ar_medium:.0%}')
    x=r.dio_change;g=r.gm_change
    level='未评估' if pd.isna(x) or pd.isna(g) else '高' if x>t.dio_high and abs(g)>t.gm_change else '中' if x>t.dio_medium or abs(g)>t.gm_change else '低'
    add('inventory','存货周转与毛利率',level,x,
        'DIO = 平均存货 / 营业成本 × 365；ΔDIO = 本年DIO − 上年DIO；GM = 1 − 营业成本 / 营收',
        '周转变慢叠加毛利率异动，需核对跌价准备、成本结转与产品结构。需三个连续年度才能计算ΔDIO。',
        ['inventory','cogs','revenue'],[year-2,year-1,year],f'高：ΔDIO>{t.dio_high:g}天且 |ΔGM|>{t.gm_change:.0%}；中：ΔDIO>{t.dio_medium:g}天或 |ΔGM|>{t.gm_change:.0%}')
    for code,title,col,formula,items,scale,years in [
        ('balance','资产负债勾稽','balance_gap','资产总计 − 负债合计 − 所有者权益合计',['assets','liabilities','equity'],r.assets,[year]),
        ('cash_bridge','现金流三段勾稽','cash_bridge_gap','期末现金 − 期初现金 − CFO − CFI − CFF − 汇率影响',['cash_end','cash_begin','cfo','cfi','cff','fx_effect'],r.cash_end,[year]),
        ('cash_continuity','跨期现金衔接','cash_continuity_gap','本期期初现金 − 上期期末现金',['cash_begin','cash_end'],r.cash_begin,[year-1,year])]:
        x=r[col];tol=max(.01,abs(scale)*t.balance_tolerance) if pd.notna(scale) else .01
        add(code,title,'未评估' if pd.isna(x) else '高' if abs(x)>tol else '低',x,formula,
            '超出容差需复核数据抽取、报表范围及重述；不能仅凭差额判定造假。',items,years,f'绝对差额容差 {tol:.3f} 百万元')
    return results


PEER_NUMERIC = ['price','shares','revenue_ttm','net_profit_ttm','ebitda_ttm','debt','cash',
                'minority_interest','preferred_equity','parent_equity_begin','parent_equity_end',
                'cogs_ttm','revenue_start','cagr_years']


def peer_matrix(peers: pd.DataFrame) -> pd.DataFrame:
    required=PEER_NUMERIC+['company','sector','currency','unit','as_of','ttm_end','scope']
    missing=set(required)-set(peers.columns)
    if missing: raise ValueError('可比数据缺少：'+', '.join(sorted(missing)))
    d=peers.copy()
    if d.empty or d.company.duplicated().any(): raise ValueError('可比公司不能为空或重复。')
    for key in PEER_NUMERIC: d[key]=d[key].map(number)
    for key,value in [('currency','CNY'),('unit','million'),('scope','consolidated')]:
        if not d[key].eq(value).all(): raise ValueError('可比数据币种、单位、合并口径须一致。')
    for k in ['as_of','ttm_end','sector']:
        if d[k].nunique(dropna=False)!=1: raise ValueError(f'可比数据 {k} 不一致。')
    if pd.to_datetime(d.as_of,errors='coerce').isna().any() or pd.to_datetime(d.ttm_end,errors='coerce').isna().any():
        raise ValueError('估值日期或 TTM 日期非法。')
    if (pd.to_datetime(d.ttm_end)>pd.to_datetime(d.as_of)).any(): raise ValueError('TTM 不能晚于估值日。')
    if (d[['price','shares']]<=0).any().any() or d[['price','shares']].isna().any().any(): raise ValueError('股价和股数必须为正。')
    d['market_cap']=d.price*d.shares
    # EBITDA 合并口径对应加入少数股东权益及优先股的企业价值。
    d['enterprise_value']=d.market_cap+d.debt+d.minority_interest+d.preferred_equity-d.cash
    for col,num,den in [('pe','market_cap','net_profit_ttm'),('ps','market_cap','revenue_ttm'),('ev_ebitda','enterprise_value','ebitda_ttm')]:
        d[col]=[ratio(n,v) if pd.notna(n) and n>0 else np.nan for n,v in zip(d[num],d[den])]
    d['roe']=[ratio(n,(a+b)/2) for n,a,b in zip(d.net_profit_ttm,d.parent_equity_begin,d.parent_equity_end)]
    d['gross_margin']=[1-ratio(c,r) for c,r in zip(d.cogs_ttm,d.revenue_ttm)]
    d['revenue_cagr']=[(ratio(end,start)**(1/years)-1) if start>0 and end>0 and years>0 else np.nan
                       for end,start,years in zip(d.revenue_ttm,d.revenue_start,d.cagr_years)]
    return d


def valuation_scenarios(df: pd.DataFrame, peers: pd.DataFrame, shares: float = 120,
                        method: str = 'P/E', growth: float = .05, spread: float = .15) -> pd.DataFrame:
    """同业分位数×目标公司一期预测因子。用户假设；不是市场预测。"""
    if not (np.isfinite(shares) and shares>0): raise ValueError('总股本（百万股）须为正。')
    if not (-.8<=growth<=1 and 0<=spread<=.5 and growth-spread>-1): raise ValueError('情景假设超出有效范围。')
    r=validate_financials(df).iloc[-1];pm=peer_matrix(peers)
    if str(pm.ttm_end.iloc[0])!=f'{int(r.year)}-12-31': raise ValueError('目标财年与同业 TTM 不一致。请提供匹配同业数据。')
    mapping={'P/E':('pe','parent_net_profit'),'P/S':('ps','revenue'),'EV/EBITDA':('ev_ebitda','ebitda')}
    if method not in mapping: raise ValueError('不支持的估值方法。')
    col,base=mapping[method];valid=pm[col].dropna()
    if len(valid)<3: raise ValueError('至少需要 3 家有效可比公司。')
    if pd.isna(r[base]) or r[base]<=0: raise ValueError(f'{LABELS[base]} 非正或缺失，此方法不适用。')
    bridge=0.0
    if method=='EV/EBITDA':
        if r[['debt','cash','minority_interest','preferred_equity']].isna().any(): raise ValueError('EV 转股权价值所需科目缺失。')
        bridge=r.debt+r.minority_interest+r.preferred_equity-r.cash
    rows=[]
    for label,q,g in [('熊市',.25,growth-spread),('基准',.5,growth),('牛市',.75,growth+spread)]:
        multiple=float(valid.quantile(q));factor=float(r[base])*(1+g);equity=factor*multiple-bridge
        rows.append(dict(scenario=label,multiple=multiple,factor=factor,growth=g,
                         equity_value=equity,target_price=equity/shares if equity>0 else np.nan,
                         method=method,valid_peers=len(valid),shares=shares,bridge=float(bridge)))
    return pd.DataFrame(rows)
