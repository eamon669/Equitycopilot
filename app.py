"""Equity Research & Due Diligence Copilot — Streamlit 入口。"""
from __future__ import annotations
# A direct double-click must launch Streamlit, not execute a bare app script.
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        _inside_streamlit = get_script_run_ctx(suppress_warning=True) is not None
    except ImportError:
        _inside_streamlit = False
    if not _inside_streamlit:
        from launcher import entrypoint
        raise SystemExit(entrypoint())

from dataclasses import asdict
from hashlib import sha256
from html import escape
from io import BytesIO
import json
import logging
import time
import uuid
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from financial_engine import (Dataset,Thresholds,parse_csv,parse_pdf,calculate_metrics,
    audit_financials,peer_matrix,valuation_scenarios,MAX_BYTES)
from mock_data import (COMPANIES,LABELS,STATEMENTS,demo_financials,demo_documents,demo_peers,financials_to_long)
from agent_copilot import (build_corpus,extract_risks,answer_question,generate_memo,memo_to_pdf,pdf_compatibility_bundle,fmt,source_label)

st.set_page_config(page_title='Equity Copilot | 智能投研终端',page_icon='◈',layout='wide',initial_sidebar_state='expanded')
logging.basicConfig(level=logging.WARNING,format='%(levelname)s %(name)s %(message)s')
logger=logging.getLogger('equity_copilot')
CSS='''
<style>
:root{--bg:#0B0F19;--panel:#111827;--line:#283247;--mute:#8D9BB3;--red:#EF4444;--blue:#2563EB;}
.stApp{background:#0B0F19;color:#E5ECF6;}
.block-container{padding:1.15rem 1.7rem 2rem;max-width:1800px;}
[data-testid="stSidebar"]{background:#0E1523;border-right:1px solid #283247;}
[data-testid="stSidebar"] .block-container{padding-top:1rem;}
h1,h2,h3{letter-spacing:-.02em;}h3{font-size:1.05rem!important;padding:.25rem 0 .45rem!important;}
[data-testid="stMetric"]{background:#111827;border:1px solid #283247;border-top:2px solid #335683;border-radius:5px;padding:12px 14px;}
[data-testid="stMetricValue"]{font-family:Consolas,'SFMono-Regular',monospace;font-size:1.7rem;color:#F4F7FC;}
[data-testid="stMetricLabel"]{color:#9EACC2;font-size:.8rem;}
[data-testid="stVerticalBlock"]{gap:.7rem;}
.stTabs [data-baseweb="tab-list"]{gap:16px;border-bottom:1px solid #283247;}
.stTabs [data-baseweb="tab"]{height:43px;padding:0 3px;font-size:.9rem;}
.stTabs [aria-selected="true"]{color:#7EAEFF;}
[data-testid="stExpander"]{background:#101725;border:1px solid #283247;border-radius:5px;}
[data-testid="stForm"]{border:1px solid #283247;padding:12px;}
[data-testid="stDataFrame"]{border:1px solid #283247;}
.brand{font-size:12px;letter-spacing:.18em;font-weight:700;color:#88A2C6;margin-bottom:9px;}
.headline{font-size:28px;line-height:1.22;font-weight:730;margin:0 0 8px;letter-spacing:-.04em;}
.subhead{font-size:12px;color:#91A1BC;line-height:1.7;}
.tag{display:inline-block;border:1px solid #345074;background:#11233C;color:#A8C9FC;padding:2px 7px;border-radius:3px;font-size:11px;}
.signal{padding:12px 14px;border:1px solid #283247;border-left:3px solid var(--risk);background:#111827;border-radius:4px;min-height:120px;}
.signal .kind{font-size:12px;color:#AAB7CC;}.signal .value{font:600 27px Consolas,monospace;color:var(--risk);margin:5px 0;}
.signal .foot{font-size:11px;color:#7F8FA8;line-height:1.5;}
.section-tag{font-size:11px;letter-spacing:.12em;color:#7D90AD;margin:5px 0;}
.rail-title{font-size:20px;font-weight:750;color:#E7EFFA;line-height:1.4;}
.rail-note{color:#8294AE;font-size:12px;line-height:1.7;}
@media(max-width:800px){.block-container{padding:1rem .7rem}.headline{font-size:22px}}
</style>
'''
st.markdown(CSS,unsafe_allow_html=True)
if 'session_token' not in st.session_state:st.session_state.session_token=uuid.uuid4().hex
if 'thresholds' not in st.session_state:st.session_state.thresholds=asdict(Thresholds())

# 对上传财务与文本的缓存键加入会话随机量，避免不同会话命中同一机密数据缓存。
@st.cache_data(ttl=1800,max_entries=16,show_spinner=False)
def cached_csv(content: bytes,name: str,session_token: str):return parse_csv(content,name)

@st.cache_data(ttl=1800,max_entries=8,show_spinner=False)
def cached_pdf(content: bytes,name: str,company: str,session_token: str):return parse_pdf(content,name,company)

@st.cache_data(ttl=1800,max_entries=24,show_spinner=False)
def cached_analysis(df,provenance,documents,thresholds,session_token):
    return (calculate_metrics(df),audit_financials(df,provenance,Thresholds(**thresholds)),
            build_corpus(df,provenance,documents),extract_risks(documents))

@st.cache_data(ttl=1800,max_entries=24,show_spinner=False)
def cached_peers(peers,session_token):return peer_matrix(peers)

@st.cache_data(ttl=1800,max_entries=48,show_spinner=False)
def cached_valuation(df,peers,shares,method,growth,spread,session_token):
    return valuation_scenarios(df,peers,shares,method,growth,spread)


def activate(dataset: Dataset, label: str, peers: pd.DataFrame, is_demo: bool):
    st.session_state.active=dict(dataset=dataset,label=label,peers=peers,is_demo=is_demo,
                                 id=uuid.uuid4().hex)
    # 原子提交后清除旧主体的对话、估值与报告，避免跨公司污染。
    for key in ['messages','scenario','memo','memo_pdf','memo_snapshot','memo_pdf_compatible','memo_pdf_preview','memo_pdf_pages','scenario_error','pending_import','pending_text']:
        st.session_state.pop(key,None)


def activate_demo(anomalous=True):
    df=demo_financials(anomalous);filename='demo_anomalous.csv' if anomalous else 'demo_healthy.csv'
    ds=parse_csv(financials_to_long(df).to_csv(index=False).encode('utf-8-sig'),filename)
    ds.documents=demo_documents(anomalous)
    activate(ds,'虚构演示数据 · 2025年末快照',demo_peers(),True)


if 'active' not in st.session_state:activate_demo()


def chart_style(fig,height=290):
    fig.update_layout(template='plotly_dark',height=height,paper_bgcolor='#0B0F19',plot_bgcolor='#0B0F19',
        margin=dict(l=8,r=8,t=25,b=15),font=dict(color='#9FAFC5',size=11),
        legend=dict(orientation='h',y=1.14,x=0),hovermode='x unified')
    fig.update_xaxes(showgrid=False);fig.update_yaxes(gridcolor='#202A3D',zerolinecolor='#283247')
    return fig


def import_controls():
    with st.expander('数据工作台  /  切换演示 · 导入财报与招股书',expanded=False):
        a,b=st.columns([1,3])
        with a:
            st.caption('内置对比样本')
            with st.form('demo_form'):
                sample=st.selectbox('演示公司',list(COMPANIES))
                apply=st.form_submit_button('加载演示样本',use_container_width=True)
            if apply:activate_demo(sample.startswith('异常'));st.rerun()
            st.download_button('下载财务 CSV 模板',financials_to_long(demo_financials()).to_csv(index=False).encode('utf-8-sig'),
                               'financial_template.csv','text/csv',use_container_width=True)
            st.download_button('下载同业 CSV 模板',demo_peers().to_csv(index=False).encode('utf-8-sig'),
                               'peers_template.csv','text/csv',use_container_width=True)
        with b:
            with st.form('import_form'):
                company=st.text_input('上传资料所属公司（须与 CSV 主体一致）',placeholder='例如：某某股份有限公司')
                left,right=st.columns(2)
                csv_file=left.file_uploader('标准化财务 CSV',type=['csv'])
                pdf_file=right.file_uploader('年度报告 / 招股书 PDF',type=['pdf'])
                peer_file=st.file_uploader('匹配财年和行业的同业 CSV（可选）',type=['csv'])
                confirm=st.checkbox('我已确认资料属于同一主体；使用 PDF 候选财务值时会先逐项复核')
                submit=st.form_submit_button('解析并准备导入',type='primary')
            if submit:
                st.session_state.pop('pending_import',None)
                st.session_state.pop('pending_text',None)
                try:
                    if not company.strip() or not confirm:raise ValueError('请填写公司并确认资料归属。')
                    if not csv_file and not pdf_file:raise ValueError('至少上传 CSV 或 PDF。')
                    for f in [csv_file,pdf_file,peer_file]:
                        if f and f.size>MAX_BYTES:raise ValueError('单个文件不得超过 20 MB。')
                    salt=st.session_state.session_token
                    ds=cached_csv(csv_file.getvalue(),csv_file.name,salt) if csv_file else Dataset(pd.DataFrame(),[],[],[])
                    if not ds.financials.empty and ds.financials.company.iloc[0]!=company.strip():raise ValueError('CSV 主体与输入公司不一致。')
                    # 深缓存返回副本；文档与财务来源始终区分。CSV 显式优先，不拼接 PDF 候选值。
                    pdf_ds=cached_pdf(pdf_file.getvalue(),pdf_file.name,company.strip(),salt) if pdf_file else None
                    if pdf_ds:
                        ds.documents=pdf_ds.documents;ds.warnings+=pdf_ds.warnings
                        if ds.financials.empty:ds.financials=pdf_ds.financials;ds.provenance=pdf_ds.provenance
                    peers=pd.DataFrame()
                    if peer_file:
                        peers=pd.read_csv(BytesIO(peer_file.getvalue()));peer_matrix(peers)
                        if not ds.financials.empty and str(peers.ttm_end.iloc[0])!=f'{int(ds.financials.year.max())}-12-31':raise ValueError('同业 TTM 与目标公司最新年度不匹配。')
                    if ds.financials.empty:
                        st.session_state.pending_text=ds
                        raise ValueError('文档文本已提取，但没有可靠财务表。请补充 CSV 并连同 PDF 重新提交；下方可查看文本。')
                    if not csv_file:
                        st.session_state.pending_import=(ds,peers)
                        st.info('候选财务表已准备，请在下方复核后正式启用。')
                    else:
                        st.session_state.pop('pending_import',None);st.session_state.pop('pending_text',None)
                        activate(ds,'用户导入数据'+(' · 同业文件：'+peer_file.name if peer_file else ' · 无同业数据'),peers,False);st.rerun()
                except Exception as exc:st.error(f'导入未完成：{exc}')
            if 'pending_import' in st.session_state:
                ds,peers=st.session_state.pending_import
                st.dataframe(ds.financials,use_container_width=True)
                st.dataframe(pd.DataFrame(ds.provenance),use_container_width=True)
                for w in ds.warnings:st.warning(w)
                if st.button('已逐项复核，启用此 PDF 数据',type='primary'):
                    activate(ds,'用户 PDF 数据 · 已由用户确认',peers,False)
                    st.session_state.pop('pending_import',None);st.rerun()
                if st.button('放弃此候选数据'):st.session_state.pop('pending_import',None);st.rerun()
            if 'pending_text' in st.session_state:
                st.caption('本次 PDF 的文本提取结果（尚未启用分析）')
                st.dataframe(pd.DataFrame(st.session_state.pending_text.documents),use_container_width=True)


def sidebar():
    with st.sidebar:
        st.markdown('<div class="brand">RESEARCH SYSTEMS / 01</div><div class="rail-title">EQUITY<br>COPILOT</div>',unsafe_allow_html=True)
        st.caption('智能投研与企业尽调终端')
        st.divider()
        st.markdown('**筛查参数**')
        with st.form('threshold_form'):
            current=st.session_state.thresholds
            ch=st.number_input('现金转化率：高风险阈值',0.0,1.5,float(current['cfo_high']),.05)
            cm=st.number_input('现金转化率：中风险阈值',0.0,2.0,float(current['cfo_medium']),.05)
            ah=st.number_input('应收增速差：高风险（百分点）',1.0,100.0,float(current['ar_high']*100),1.0)
            am=st.number_input('应收增速差：中风险（百分点）',0.0,99.0,float(current['ar_medium']*100),1.0)
            submit=st.form_submit_button('应用筛查参数',use_container_width=True)
        if submit:
            try:
                t=Thresholds(cfo_high=ch,cfo_medium=cm,ar_high=ah/100,ar_medium=am/100);t.validate()
                st.session_state.thresholds=asdict(t)
                st.session_state.pop('memo',None);st.session_state.pop('memo_pdf',None)
            except ValueError as exc:st.error(str(exc))
        st.divider()
        st.markdown('**工作流**')
        st.markdown('01　导入与核实\n\n02　财务风险筛查\n\n03　同业估值\n\n04　证据问答\n\n05　投资备忘录')
        st.markdown('<div class="rail-note">默认本地处理<br>人民币 / 合并报表 / 年频<br>数据入口始终位于主区域顶部</div>',unsafe_allow_html=True)
        st.caption('PoC v1.1 · 启动与 PDF 修复版')


def audit_view(ds,metrics,audit):
    st.markdown('<div class="section-tag">FINANCIAL INTEGRITY / 可解释财务筛查</div>',unsafe_allow_html=True)
    colors={'高':'#EF4444','中':'#F59E0B','低':'#10B981','未评估':'#8D9BB3'}
    for offset in (0,3):
        cols=st.columns(3)
        for col,risk in zip(cols,audit[offset:offset+3]):
            with col:
                value=fmt(risk['value'],risk['code'] in ['cash_quality','receivables'])
                if risk['code']=='inventory' and risk['value'] is not None:value+=' 天'
                st.markdown(f'<div class="signal" style="--risk:{colors[risk["level"]]}"><div class="kind">{escape(risk["title"])} · {risk["level"]}</div><div class="value">{value}</div><div class="foot">{escape(risk["threshold"])}</div></div>',unsafe_allow_html=True)
                with st.expander('计算公式与原始来源',expanded=False):
                    st.code(risk['formula'],language=None,wrap_lines=True)
                    st.write(risk['reason'])
                    st.dataframe(pd.DataFrame(risk['inputs']),hide_index=True,use_container_width=True)
                    if risk['sources']:st.dataframe(pd.DataFrame(risk['sources']),hide_index=True,use_container_width=True)
                    else:st.caption('未取得对应来源，不能视作已验证。')
    left,right=st.columns(2)
    with left:
        st.markdown('### 利润与现金流质量')
        fig=go.Figure()
        for key,name,color in [('net_profit','净利润','#568FFF'),('cfo','经营现金流','#10B981')]:
            fig.add_bar(x=metrics.year.astype(str),y=metrics[key],name=name,marker_color=color)
        fig.update_layout(barmode='group',yaxis_title='人民币百万元')
        st.plotly_chart(chart_style(fig),use_container_width=True,key='cash_chart')
    with right:
        st.markdown('### 营运资金与毛利率')
        fig=go.Figure()
        fig.add_bar(x=metrics.year.astype(str),y=metrics.inventory,name='存货',marker_color='#345174')
        fig.add_bar(x=metrics.year.astype(str),y=metrics.receivables,name='应收账款',marker_color='#647DA4')
        fig.add_scatter(x=metrics.year.astype(str),y=metrics.gross_margin*100,name='毛利率',mode='lines+markers',yaxis='y2',line=dict(color='#F59E0B',width=2))
        fig.update_layout(yaxis_title='百万元',yaxis2=dict(overlaying='y',side='right',title='毛利率 %',showgrid=False))
        st.plotly_chart(chart_style(fig),use_container_width=True,key='working_capital')
    st.caption('规则阈值可调整。正净利润下 CFO/净利润才适用；DIO 使用平均存货；勾稽差异先排查抽取和重述。')


@st.fragment
def valuation_view(active):
    peers=active['peers'];ds=active['dataset'];salt=st.session_state.session_token
    if peers.empty:
        st.info('未导入同业数据。请在数据工作台使用同业模板，导入匹配年度、币种与行业的数据。');return
    try:pm=cached_peers(peers,salt)
    except Exception as exc:st.error(f'同业数据不可用：{exc}');return
    st.markdown('### 同业可比矩阵')
    st.caption(f'{len(pm)} 家 · {pm.sector.iloc[0]} · 估值日 {pm.as_of.iloc[0]} · TTM {pm.ttm_end.iloc[0]} · '+('全部虚构数据' if active['is_demo'] else '用户提供数据'))
    columns=['company','price','pe','ps','ev_ebitda','roe','gross_margin','revenue_cagr']
    display=pm[columns].copy()
    for c in ['roe','gross_margin','revenue_cagr']:display[c]*=100
    st.dataframe(display,hide_index=True,use_container_width=True,column_config={
        'company':'可比公司','price':st.column_config.NumberColumn('股价 / 元',format='%.2f'),
        'pe':st.column_config.NumberColumn('P/E · TTM',format='%.2f ×'),
        'ps':st.column_config.NumberColumn('P/S · TTM',format='%.2f ×'),
        'ev_ebitda':st.column_config.NumberColumn('EV/EBITDA',format='%.2f ×'),
        **{c:st.column_config.NumberColumn(label,format='%.1f%%') for c,label in [('roe','ROE'),('gross_margin','毛利率'),('revenue_cagr','营收 CAGR')]}})
    left,right=st.columns([1.15,1])
    with left:
        st.markdown('### 盈利质量 × 估值溢价')
        chart=pm.dropna(subset=['pe','roe']).copy();chart['ROE %']=chart.roe*100
        fig=px.scatter(chart,x='ROE %',y='pe',size='market_cap',hover_name='company',color='gross_margin',
                       color_continuous_scale=['#32507C','#6D9FF5'],labels={'pe':'P/E (TTM)','gross_margin':'毛利率'})
        fig.update_traces(marker=dict(line=dict(width=1,color='#A4C9FF')))
        st.plotly_chart(chart_style(fig,360),use_container_width=True,key='scatter')
    with right:
        st.markdown('### 情景假设与目标价')
        with st.form('valuation_form_'+active['id']):
            c1,c2=st.columns(2)
            method=c1.selectbox('估值方法',['P/E','P/S','EV/EBITDA'])
            shares=c2.number_input('总股本 / 百万股',min_value=.001,value=120.0,step=10.0)
            growth=st.slider('基准因子增长假设',-50,60,5,1,format='%d%%')
            spread=st.slider('牛熊增长偏移',0,30,15,1,format='%d%%')
            submit=st.form_submit_button('计算估值区间',type='primary',use_container_width=True)
        if submit or ('scenario' not in st.session_state and active['is_demo']):
            st.session_state.pop('scenario',None);st.session_state.pop('memo',None);st.session_state.pop('memo_pdf',None)
            try:
                result=cached_valuation(ds.financials,peers,shares,method,growth/100,spread/100,salt)
                st.session_state.scenario=result;st.session_state.pop('scenario_error',None)
            except Exception as exc:st.session_state.scenario_error=str(exc)
        if 'scenario_error' in st.session_state:st.warning(st.session_state.scenario_error)
        if 'scenario' in st.session_state:
            result=st.session_state.scenario
            for c,(_,row) in zip(st.columns(3),result.iterrows()):c.metric(row.scenario+' / 元',fmt(row.target_price))
            st.caption(f'已应用：{result.method.iloc[0]} · 股本 {result.shares.iloc[0]:g} 百万股；同业 25% / 50% / 75% 分位数。')
            st.dataframe(result[['scenario','growth','multiple','factor','equity_value','target_price']],hide_index=True,use_container_width=True)
    with st.expander('估值口径与公式'):
        st.write('P/E = 市值 / TTM归母净利润；P/S = 市值 / TTM营业收入；EV = 市值 + 有息负债 + 少数股东权益 + 优先股权益 − 非受限现金。')
        st.write('ROE = TTM归母净利润 / 平均归母权益；CAGR = (期末收入 / 期初收入)^(1/年数) − 1。非正分母显示 N/A，并从倍数分位数中剔除。')
        st.write('目标价 = 预测因子 × 可比倍数 / 总股本；EV/EBITDA 先扣除有息负债、少数股东权益与优先股权益，再加现金。三种方法单独估值，不混合平均。')
        st.write('一期预测使用用户增长假设。财年相同不代表披露已于估值日公开；本 PoC 不用于历史回测。')


@st.fragment
def agent_view(active,corpus,risks):
    st.markdown('### 披露风险抽取')
    risk_df=pd.DataFrame(risks)[['topic','level','value','text','source','page','item']]
    st.dataframe(risk_df,hide_index=True,use_container_width=True,column_config={'topic':'类别','level':'信号','text':'披露原文','source':'文件','page':'Page','item':'Item','value':st.column_config.NumberColumn('集中度（小数）',format='%.3f')})
    st.caption('句级规则候选；否定表述单独处理。未检出不代表不存在。完整审计意见与诉讼状态需复核原文。')
    left,right=st.columns([1.55,1])
    with left:
        st.markdown('### NL2Finance / 证据问答')
        with st.form('question_form',clear_on_submit=False):
            question=st.text_area('向当前公司提问',value='分析近三年毛利率变化及披露中的主因',height=90,max_chars=2000)
            remote=st.checkbox('使用已配置的大模型：将检索片段发送至 OpenAI',value=False)
            submit=st.form_submit_button('检索并回答',type='primary')
        if submit:
            start=time.perf_counter()
            try:
                with st.spinner('正在核对证据…'):
                    answer=answer_question(question,active['dataset'].financials,corpus,remote)
                answer['elapsed']=time.perf_counter()-start
                history=st.session_state.get('messages',[])
                st.session_state.messages=(history+[{'question':question,'answer':answer}])[-10:]
            except Exception as exc:st.error(f'无法完成问答：{exc}')
        for entry in st.session_state.get('messages',[]):
            with st.chat_message('user'):st.write(entry['question'])
            with st.chat_message('assistant'):
                st.markdown(entry['answer']['text'])
                st.caption(f'{entry["answer"]["mode"]} · {entry["answer"]["elapsed"]:.2f}s')
                if entry['answer'].get('notice'):st.info(entry['answer']['notice'])
    with right:
        st.markdown('### Evidence / 引用证据')
        history=st.session_state.get('messages',[])
        if not history:st.info('提交问题后，在这里查看引用 ID、Page/Item 和对应原文。')
        else:
            for c in history[-1]['answer']['sources']:
                with st.expander(f'[{c["id"]}] {c["item"]}',expanded=True):
                    st.text(source_label(c));st.write(c['text'])
        if st.button('清空当前公司对话'):
            st.session_state.messages=[];st.rerun(scope='fragment')


@st.fragment
def memo_view(active, audit):
    st.markdown('### C-Level 投资备忘录')
    st.write('根据当前数据生成报告，导出前逐页检查 PDF 是否有可见内容。')
    st.caption('兼容版为完整页面图像；可检索版保留可复制文字。两者均来自当前报告。')
    if st.button('生成投资备忘录', type='primary'):
        for key in ['memo','memo_pdf','memo_pdf_compatible','memo_pdf_preview','memo_pdf_pages']:
            st.session_state.pop(key, None)
        try:
            ds = active['dataset']
            pm = cached_peers(active['peers'], st.session_state.session_token) if not active['peers'].empty else pd.DataFrame()
            md = generate_memo(ds.financials, audit, pm, st.session_state.get('scenario'),
                               ds.documents, ds.provenance, active['label'])
            st.session_state.memo = md
            try:
                with st.spinner('生成并检查 PDF 页面…'):
                    standard = memo_to_pdf(md)
                    bundle = pdf_compatibility_bundle(standard)
                st.session_state.memo_pdf = standard
                st.session_state.memo_pdf_compatible = bundle['compatible_pdf']
                st.session_state.memo_pdf_preview = bundle['preview_png']
                st.session_state.memo_pdf_pages = bundle['page_count']
            except Exception as exc:
                st.error(f'PDF 未通过导出检查：{exc}。Markdown 仍可下载。')
        except Exception as exc:
            st.error(f'报告生成失败：{exc}')
    if 'memo' not in st.session_state:
        st.info('点击生成后，将在此处展示实际 PDF 第一页和下载按钮。')
        return
    c1, c2, c3 = st.columns(3)
    c1.download_button('下载 Markdown', st.session_state.memo.encode('utf-8'),
                       'investment_memo.md', 'text/markdown', use_container_width=True, on_click='ignore')
    if 'memo_pdf_compatible' in st.session_state:
        c2.download_button('下载兼容版 PDF', st.session_state.memo_pdf_compatible,
                           'investment_memo_compatible.pdf', 'application/pdf',
                           use_container_width=True, on_click='ignore')
        c3.download_button('下载可检索版 PDF', st.session_state.memo_pdf,
                           'investment_memo_searchable.pdf', 'application/pdf',
                           use_container_width=True, on_click='ignore')
        st.success(f'PDF 共 {st.session_state.memo_pdf_pages} 页，已完成渲染与非空检查。')
        with st.expander('实际 PDF 页面预览', expanded=True):
            st.image(st.session_state.memo_pdf_preview, caption='这是下载文件对应的第一页', use_container_width=True)
    with st.expander('查看 Markdown 全文', expanded=False):
        st.markdown(st.session_state.memo)


def source_view(ds,metrics):
    st.markdown('### 标准化财务三表')
    st.caption('金额：人民币百万元；合并年报。空值保持 N/A。PDF 财务值需在导入时复核。')
    cols=st.columns(3)
    for col,statement,label in zip(cols,['BS','IS','CF'],['资产负债表','利润表','现金流量表']):
        with col:
            st.markdown('**'+label+'**')
            keys=[k for k,v in STATEMENTS.items() if v==statement]
            table=ds.financials.set_index('year')[keys].rename(columns=LABELS).T
            table.columns=table.columns.astype(str)
            st.dataframe(table.style.format('{:,.2f}',na_rep='N/A'),use_container_width=True,height=450)
    with st.expander('全量来源索引',expanded=False):st.dataframe(pd.DataFrame(ds.provenance),hide_index=True,use_container_width=True)
    with st.expander('文档分块与 Page/Item',expanded=False):st.dataframe(pd.DataFrame(ds.documents),hide_index=True,use_container_width=True)
    st.download_button('导出标准化财务数据',financials_to_long(ds.financials).to_csv(index=False).encode('utf-8-sig'),'normalized_financials.csv','text/csv')


def main():
    started=time.perf_counter();sidebar()
    st.markdown('<div class="brand">EQUITY RESEARCH & DUE DILIGENCE / COPILOT</div>',unsafe_allow_html=True)
    import_controls()
    active=st.session_state.active;ds=active['dataset']
    metrics,audit,corpus,risks=cached_analysis(ds.financials,ds.provenance,ds.documents,st.session_state.thresholds,st.session_state.session_token)
    r=metrics.iloc[-1];high=sum(x['level']=='高' for x in audit)
    left,right=st.columns([3,1])
    with left:
        st.markdown(f'<div class="headline">{escape(str(r.company))}</div><div class="subhead">{escape(active["label"])}　<span class="tag">FY {int(r.year)}</span>　CNY million · 合并口径</div>',unsafe_allow_html=True)
    with right:
        st.metric('高风险线索 / 已评估',f'{high} / {sum(x["level"]!="未评估" for x in audit)}')
    for w in ds.warnings:st.warning(w)
    a,b,c,d=st.columns(4)
    a.metric('营业收入 / 百万元',fmt(r.revenue),fmt(r.revenue_growth,True)+' YoY' if pd.notna(r.revenue_growth) else None)
    b.metric('归母净利润 / 百万元',fmt(r.parent_net_profit))
    c.metric('毛利率',fmt(r.gross_margin,True),f'{r.gm_change*100:+.1f} pp' if pd.notna(r.gm_change) else None)
    d.metric('经营现金流 / 百万元',fmt(r.cfo))
    tabs=st.tabs(['01 财务审计','02 同业估值','03 投研 Agent','04 投资备忘录','05 三表与来源'])
    with tabs[0]:audit_view(ds,metrics,audit)
    with tabs[1]:valuation_view(active)
    with tabs[2]:agent_view(active,corpus,risks)
    with tabs[3]:memo_view(active,audit)
    with tabs[4]:source_view(ds,metrics)
    st.caption(f'本次服务端页面构建 {time.perf_counter()-started:.2f}s · 不含浏览器渲染 / 网络传输 · 风险线索须由分析师复核')


try:main()
except Exception as exc:
    # 顶层兜底保持入口可见；日志仅记录错误类型，不包含财报正文或密钥。
    logger.error('App failed: %s',type(exc).__name__)
    st.error(f'页面暂时无法完成分析（{type(exc).__name__}）。请重新加载演示或检查导入格式。')
    if st.button('恢复默认演示'):activate_demo();st.rerun()
