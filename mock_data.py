"""固定随机种子的虚构人民币财报；所有金额为百万元，股数为百万股。"""
from __future__ import annotations
import pandas as pd
import numpy as np

LABELS = {
    'revenue': '营业收入', 'cogs': '营业成本', 'net_profit': '净利润',
    'parent_net_profit': '归母净利润', 'ebitda': 'EBITDA',
    'assets': '资产总计', 'liabilities': '负债合计', 'equity': '所有者权益合计',
    'parent_equity': '归母所有者权益', 'receivables': '应收账款', 'inventory': '存货',
    'cash': '非受限现金', 'debt': '有息负债', 'minority_interest': '少数股东权益',
    'preferred_equity': '优先股权益', 'cfo': '经营活动现金流量净额',
    'cfi': '投资活动现金流量净额', 'cff': '筹资活动现金流量净额',
    'fx_effect': '汇率变动影响', 'cash_begin': '期初现金及现金等价物',
    'cash_end': '期末现金及现金等价物',
}
STATEMENTS = {k: ('IS' if k in ['revenue','cogs','net_profit','parent_net_profit','ebitda']
                    else 'CF' if k in ['cfo','cfi','cff','fx_effect','cash_begin','cash_end'] else 'BS')
              for k in LABELS}
COMPANIES = {'异常样本 · 星衡智造': '星衡智造（虚构）', '健康样本 · 澄川工业': '澄川工业（虚构）'}


def demo_financials(anomalous: bool = True) -> pd.DataFrame:
    """2022 年为期初比较年；2023—2025 年可计算平均余额类指标。"""
    rows = []
    company = COMPANIES['异常样本 · 星衡智造' if anomalous else '健康样本 · 澄川工业']
    for i, year in enumerate(range(2022, 2026)):
        revenue = [1000,1200,1440,1656][i]
        cogs = [680,840,1051.2,1291.68][i] if anomalous else revenue * .68
        profit = [90,105,120,155][i] if anomalous else [90,108,129.6,149.04][i]
        cfo = [100,105,68,-22][i] if anomalous else profit * 1.15
        inventory = [140,180,280,470][i] if anomalous else cogs * .20
        receivables = [160,195,320,560][i] if anomalous else revenue * .16
        assets, liabilities = [1800,2100,2480,2980][i], [850,980,1190,1510][i]
        equity = assets - liabilities + (90 if anomalous and i == 3 else 0)
        cash_begin = 200 if i == 0 else rows[-1]['cash_end']
        cfi, cff, fx = -80-i*10, 60+i*15, 2
        cash_end = cash_begin + cfo + cfi + cff + fx + (30 if anomalous and i == 3 else 0)
        row = dict(company=company,year=year,period='FY',currency='CNY',unit='million',
                   scope='consolidated',revenue=revenue,cogs=cogs,net_profit=profit,
                   parent_net_profit=profit*.96,ebitda=profit+70,assets=assets,
                   liabilities=liabilities,equity=equity,parent_equity=equity-40,
                   receivables=receivables,inventory=inventory,cash=cash_end-10,
                   debt=500+i*50,minority_interest=40,preferred_equity=0,cfo=cfo,
                   cfi=cfi,cff=cff,fx_effect=fx,cash_begin=cash_begin,cash_end=cash_end)
        rows.append(row)
    return pd.DataFrame(rows)


def financials_to_long(df: pd.DataFrame, source: str = '模拟财报.csv') -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for item in LABELS:
            if item in row and pd.notna(row[item]):
                rows.append(dict(company=row['company'],year=int(row['year']),period='FY',
                    currency='CNY',unit='million',scope='consolidated',statement=STATEMENTS[item],
                    item=item,value=float(row[item]),source=source))
    return pd.DataFrame(rows)


def demo_documents(anomalous: bool = True) -> list[dict]:
    """Page 为内置虚构文档页码，不对应任何真实发行人披露。"""
    if anomalous:
        texts = [
            (8,'公司概况','公司为工业自动化设备制造商，主要产品包括控制器及自动化产线。以下均为虚构演示资料。'),
            (42,'管理层讨论与分析','2023年至2025年，毛利率持续下滑。管理层解释：原材料采购价格上涨、低毛利集成业务收入占比提高，以及为进入新客户供应链给予价格折让。公司未披露各因素量化贡献，无法据此建立精确归因桥。'),
            (43,'营运资金','2025年部分客户验收及回款周期延长，应收账款上升；备货增加及滞销产品积压导致存货增加。需复核期后回款、收入截止及存货跌价准备。'),
            (68,'重大诉讼','截至2025年12月31日，公司存在一项未决重大诉讼，涉及合同纠纷，索赔金额人民币3,800万元。管理层认为结果仍存在不确定性，尚未收到终审判决。'),
            (72,'客户集中度','2025年前五大客户销售额占营业收入比例为64.5%，其中第一大客户占比为28.0%。客户关系变化可能影响收入稳定性。'),
            (73,'供应商集中度','2025年前五大供应商采购额占采购总额比例为57.2%。核心原材料供应商集中度较高。'),
            (96,'审计意见','审计师对2025年度财务报表发表保留意见。形成保留意见的基础：无法就部分期末存货取得充分、适当的审计证据；相关事项可能影响存货计价及利润。'),
        ]
    else:
        texts = [
            (8,'公司概况','公司为工业自动化设备制造商，服务分散的制造业客户。以下均为虚构演示资料。'),
            (42,'管理层讨论与分析','2023年至2025年毛利率维持32%。管理层表示产品组合稳定，成本管理抵消了原材料波动。'),
            (68,'重大诉讼','截至2025年12月31日，公司不存在重大诉讼或仲裁事项。'),
            (72,'客户集中度','2025年前五大客户销售额占比为31.5%，第一大客户占比为9.2%。'),
            (73,'供应商集中度','2025年前五大供应商采购额占比为29.0%。'),
            (96,'审计意见','审计师对2025年度财务报表发表标准无保留意见。'),
        ]
    return [dict(id=f'D{i+1}',source='虚构年度报告（演示文本）',page=p,item=item,text=t)
            for i,(p,item,t) in enumerate(texts)]


def demo_peers() -> pd.DataFrame:
    rng = np.random.default_rng(20260906)
    rows = []
    for i,name in enumerate(['衡岳自动化','岚川机电','北辰智控','云际装备','远川工业','青屿科技','启元精工','至衡系统']):
        rev = float(rng.uniform(1800,5200)); profit = rev * float(rng.uniform(.07,.14))
        shares = float(rng.uniform(180,420)); pe = float(rng.uniform(14,29))
        equity = profit/float(rng.uniform(.09,.19)); minority = equity*.04
        rows.append(dict(company=name+'（虚构）',sector='工业自动化',currency='CNY',
            unit='million',as_of='2025-12-31',ttm_end='2025-12-31',scope='consolidated',
            price=profit*.96*pe/shares,shares=shares,revenue_ttm=rev,net_profit_ttm=profit*.96,
            ebitda_ttm=profit*1.65,debt=rev*.22,cash=rev*.12,minority_interest=minority,
            preferred_equity=0,parent_equity_begin=equity*.90,parent_equity_end=equity,
            cogs_ttm=rev*float(rng.uniform(.62,.74)),revenue_start=rev/float(rng.uniform(1.1,1.8)),
            cagr_years=3))
    return pd.DataFrame(rows)
