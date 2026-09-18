"""可重复的本地小样本基准；不用于宣称网络端到端 SLA。"""
from pathlib import Path
from time import perf_counter
import json
import platform
import statistics
import pandas as pd
from mock_data import demo_financials,demo_documents,demo_peers,financials_to_long
from financial_engine import parse_csv,parse_pdf,calculate_metrics,audit_financials,peer_matrix,valuation_scenarios
from agent_copilot import build_corpus,answer_question


def measure(name,func,repeats=30):
    times=[]
    for _ in range(repeats):
        t=perf_counter();func();times.append((perf_counter()-t)*1000)
    return dict(name=name,iterations=repeats,p50_ms=round(statistics.median(times),2),
                p95_ms=round(float(pd.Series(times).quantile(.95)),2),max_ms=round(max(times),2))


def main():
    df=demo_financials();raw=financials_to_long(df).to_csv(index=False).encode();ds=parse_csv(raw,'demo.csv')
    docs=demo_documents();peers=demo_peers();corpus=build_corpus(df,ds.provenance,docs)
    def pipeline():
        d=parse_csv(raw,'demo.csv');calculate_metrics(d.financials);audit_financials(d.financials,d.provenance)
        peer_matrix(peers);valuation_scenarios(d.financials,peers)
    results=[measure('CSV+财务筛查+8家同业+情景估值（无Streamlit缓存）',pipeline),
             measure('本地毛利率证据问答',lambda:answer_question('近三年毛利率变化原因',df,corpus)),
             measure('4页文本PDF解析',lambda:parse_pdf(Path('examples/anomalous_annual_report.pdf').read_bytes(),'demo.pdf','星衡智造（虚构）'),10)]
    report=dict(python=platform.python_version(),platform=platform.system(),data='4个财年/84项财务记录/8家同业/4页PDF',
                excludes='浏览器渲染、网络传输、外部LLM、大文件、并发负载与冷启动',results=results)
    Path('VALIDATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
