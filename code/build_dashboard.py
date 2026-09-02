#!/usr/bin/env python3
"""Build the self-contained SoilTrust notebook and offline HTML dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd
from plotly.offline import get_plotlyjs


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CSV = ROOT / "artifacts" / "confidence_map" / "tile_predictions.csv"
SUMMARY = ROOT / "artifacts" / "confidence_map" / "summary.json"
HTML_PATH = HERE / "soiltrust_dashboard.html"
NOTEBOOK_PATH = HERE / "soiltrust_dashboard.ipynb"


def load_data() -> tuple[list[dict], dict]:
    df = pd.read_csv(CSV)
    required = {
        "region", "sample_index", "longitude", "latitude", "mean_soc_gkg",
        "model_std_gkg", "low_confidence", "any_seed_negative", "ensemble_mean_negative",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise RuntimeError("Required dashboard fields contain nulls")
    records = df[list(required)].to_dict(orient="records")
    for row in records:
        row["sample_index"] = int(row["sample_index"])
        for key in ["low_confidence", "any_seed_negative", "ensemble_mean_negative"]:
            row[key] = bool(row[key])
    return records, json.loads(SUMMARY.read_text())


def build_html(records: list[dict], summary: dict) -> str:
    payload = json.dumps(records, separators=(",", ":"))
    threshold = summary["low_confidence_threshold_std_gkg"]
    na_median = summary["north_america"]["median_disagreement_gkg"]
    eu_median = summary["europe"]["median_disagreement_gkg"]
    ratio = eu_median / na_median
    eu_low = 100 * summary["europe"]["low_confidence_fraction"]
    plotly_js = get_plotlyjs()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SoilTrust - District Soil Intelligence</title>
<script>{plotly_js}</script>
<style>
:root{{--navy:#17365d;--blue:#285a8e;--green:#247a4a;--red:#bd3b32;--saffron:#e69b28;--ink:#1e2933;--muted:#657382;--panel:#fff;--bg:#eef2f5;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);font-family:Inter,Segoe UI,Arial,sans-serif;color:var(--ink)}}
.govbar{{height:7px;background:linear-gradient(90deg,#ef8f1f 0 33%,#fff 33% 66%,#27864b 66%)}}
header{{background:var(--navy);color:#fff;padding:22px 34px 20px;box-shadow:0 2px 8px #0003}}
header h1{{margin:0;font-size:29px;letter-spacing:.2px}} header p{{margin:7px 0 0;color:#dce8f4;font-size:16px}}
.shell{{max-width:1500px;margin:auto;padding:20px 24px 32px}}
.context{{display:flex;justify-content:space-between;gap:18px;align-items:center;margin-bottom:14px}}
.context strong{{color:var(--navy)}} .tag{{background:#e3ebf4;border:1px solid #adc0d3;border-radius:20px;padding:7px 12px;font-size:13px}}
.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}}
.kpi{{background:var(--panel);border-radius:9px;padding:15px 16px;border-left:5px solid var(--blue);box-shadow:0 1px 5px #1b31451b;min-height:105px}}
.kpi.hero{{background:#fff8e8;border-left-color:var(--saffron)}} .kpi .label{{font-size:12px;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);font-weight:700}}
.kpi .value{{font-size:25px;color:var(--navy);font-weight:750;margin-top:8px}} .kpi .sub{{font-size:12px;color:var(--muted);margin-top:5px}}
.grid{{display:grid;grid-template-columns:minmax(0,2.15fr) minmax(330px,.85fr);gap:16px}}
.panel{{background:var(--panel);border-radius:10px;box-shadow:0 1px 6px #1b31451f;border:1px solid #dce3e9;overflow:hidden}}
.panel-title{{padding:14px 17px;border-bottom:1px solid #dce3e9;font-size:16px;font-weight:750;color:var(--navy);display:flex;justify-content:space-between;align-items:center}}
.controls{{padding:13px 17px;background:#f8fafb;border-bottom:1px solid #e0e6eb;display:flex;flex-wrap:wrap;gap:18px;align-items:center}}
.control{{display:flex;align-items:center;gap:8px}} label{{font-size:12px;color:var(--muted);font-weight:750;text-transform:uppercase;letter-spacing:.4px}}
select,input{{accent-color:var(--blue)}} select{{padding:7px 28px 7px 9px;border:1px solid #b7c4cf;border-radius:5px;background:#fff;color:var(--ink)}}
#threshold{{width:210px}} #thresholdValue{{font-variant-numeric:tabular-nums;font-weight:750;color:var(--navy);min-width:68px}}
#map{{height:600px;width:100%}} .legend-note{{font-size:12px;color:var(--muted)}}
.counts{{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:14px}}
.count{{border-radius:8px;padding:13px;text-align:center}} .count.trust{{background:#e7f5ec;color:#17643a}} .count.sample{{background:#fde9e7;color:#9f2923}}
.count b{{display:block;font-size:27px}} .count span{{font-size:12px;font-weight:700}}
.area{{margin:0 14px 14px;padding:11px;background:#fff4d9;border:1px solid #e9ca7a;border-radius:7px;text-align:center;font-weight:700;color:#70500b}}
.work-controls{{padding:10px 14px;background:#f8fafb;border-bottom:1px solid #e0e6eb}}
.table-wrap{{max-height:500px;overflow:auto}} table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{position:sticky;top:0;background:#e7edf3;color:var(--navy);text-align:left;padding:9px 8px;z-index:1}} td{{padding:8px;border-bottom:1px solid #e8edf1;font-variant-numeric:tabular-nums}}
tr.impossible td{{background:#fff0ef}} .rank{{font-weight:800;color:var(--navy)}} .badge{{padding:3px 6px;border-radius:10px;background:#f6d4d1;color:#8e201a;font-size:10px;font-weight:800}}
.method{{padding:14px 17px;border-top:1px solid #e0e6eb;background:#fafcfd;color:var(--muted);font-size:12px;line-height:1.45}}
footer{{margin-top:18px;background:#fff;border:1px solid #d8e0e7;border-radius:9px;padding:17px 20px;color:#4b5966;font-size:13px;line-height:1.55}}
footer strong{{color:#94332d}} .pitch{{margin-top:10px;color:var(--navy);font-size:16px;font-weight:800}}
@media(max-width:1050px){{.kpis{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}}}
@media(max-width:600px){{.shell{{padding:12px}}header{{padding:18px}}.kpis{{grid-template-columns:1fr}}#map{{height:470px}}}}
</style></head><body><div class="govbar"></div>
<header><h1>SoilTrust - District Soil Intelligence</h1><p>The soil map that tells you where to trust it - and where to send ground samples.</p></header>
<main class="shell">
<div class="context"><div><strong>Decision question:</strong> With a limited sampling budget, where should field teams go first?</div><div class="tag">Demonstration geography • India requires local calibration</div></div>
<section class="kpis">
 <div class="kpi"><div class="label">In training region</div><div class="value">{na_median:.2f} g/kg</div><div class="sub">Median model disagreement • North America</div></div>
 <div class="kpi"><div class="label">Outside training region</div><div class="value">{eu_median:.2f} g/kg</div><div class="sub">Median model disagreement • Europe</div></div>
 <div class="kpi hero"><div class="label">Transfer warning</div><div class="value">{ratio:.1f}× worse</div><div class="sub">Confidence degrades outside the training region</div></div>
 <div class="kpi"><div class="label">Selected-region low confidence</div><div class="value" id="lowKpi">—</div><div class="sub" id="lowKpiSub">At selected threshold</div></div>
 <div class="kpi"><div class="label">Reference Europe result</div><div class="value">{eu_low:.2f}%</div><div class="sub">LOW at calibrated {threshold:.2f} g/kg threshold</div></div>
</section>
<section class="grid">
 <div class="panel">
  <div class="panel-title"><span>Tile intelligence map</span><span class="legend-note">Each point is one 1.28 km benchmark tile</span></div>
  <div class="controls">
   <div class="control"><label for="region">Region</label><select id="region"><option value="North America">North America — in training region</option><option value="Europe">Europe — outside training region</option></select></div>
   <div class="control"><label for="metric">Colour by</label><select id="metric"><option value="confidence">Confidence — green trust / red sample</option><option value="prediction">Prediction — mean SOC</option></select></div>
   <div class="control"><label for="threshold">Confidence threshold</label><input id="threshold" type="range" min="0" max="130" step="0.25" value="{threshold:.2f}"><span id="thresholdValue">{threshold:.2f} g/kg</span></div>
  </div>
  <div id="map"></div>
  <div class="method">LOW confidence is operationally defined as model disagreement above the selected threshold. A physically impossible negative prediction from any seed always overrides the slider and enters the sampling queue. Moving the slider changes prioritisation—not the underlying predictions.</div>
 </div>
 <aside class="panel">
  <div class="panel-title"><span>Sampling priority</span><span class="legend-note">Field-team worklist</span></div>
  <div class="counts"><div class="count trust"><b id="trustCount">—</b><span>TRUST THE MODEL HERE</span></div><div class="count sample"><b id="sampleCount">—</b><span>SEND A GROUND TEAM</span></div></div>
  <div class="area" id="sampleArea">—% of represented tile area needs sampling</div>
  <div class="work-controls"><label for="topN">Show worklist</label> <select id="topN"><option>10</option><option selected>25</option><option>50</option><option>100</option></select></div>
  <div class="table-wrap"><table><thead><tr><th>#</th><th>Tile</th><th>Latitude</th><th>Longitude</th><th>Disagreement</th><th>Flag</th></tr></thead><tbody id="worklist"></tbody></table></div>
 </aside>
</section>
<footer><strong>Important limitation:</strong> these predictions are uncalibrated and some raw seed outputs are physically impossible. That is exactly why the confidence layer exists: disagreement and impossible values direct scarce ground-sampling resources instead of hiding uncertainty. Production deployment in India requires India ground observations—including calibration pathways using the country’s approximately 25 crore Soil Health Cards—and verified parcel boundaries.
<div class="pitch">The only soil map that tells you where to trust it, and where to send ground samples instead.</div></footer>
</main>
<script>
const ALL={payload};
const bounds={{"North America":{{x:[-175,-45],y:[20,75]}},"Europe":{{x:[-15,45],y:[32,73]}}}};
const regionEl=document.getElementById('region'), metricEl=document.getElementById('metric'), thresholdEl=document.getElementById('threshold'), topNEl=document.getElementById('topN');
const thresholdValue=document.getElementById('thresholdValue'), trustCount=document.getElementById('trustCount'), sampleCount=document.getElementById('sampleCount'), sampleArea=document.getElementById('sampleArea'), lowKpi=document.getElementById('lowKpi'), lowKpiSub=document.getElementById('lowKpiSub'), worklist=document.getElementById('worklist');
function selected(){{return ALL.filter(d=>d.region===regionEl.value)}}
function sendTeam(d,t){{return d.model_std_gkg>t || d.any_seed_negative || d.ensemble_mean_negative}}
function hover(d){{return `<b>Tile ${{d.sample_index}}</b><br>Lat ${{d.latitude.toFixed(4)}}°, Lon ${{d.longitude.toFixed(4)}}°<br>Mean SOC: ${{d.mean_soc_gkg.toFixed(2)}} g/kg<br>Disagreement: ${{d.model_std_gkg.toFixed(2)}} g/kg<br>Any negative seed: ${{d.any_seed_negative?'YES':'No'}}`}}
function trace(rows,name,metric,t,isSample){{
 const confidence=metric==='confidence';
 return {{type:'scattergl',mode:'markers',name,x:rows.map(d=>d.longitude),y:rows.map(d=>d.latitude),text:rows.map(hover),hovertemplate:'%{{text}}<extra></extra>',
 marker:{{size:isSample?11:9,opacity:.88,color:rows.map(d=>confidence?d.model_std_gkg:d.mean_soc_gkg),colorscale:confidence?'RdYlGn':'Viridis',reversescale:confidence,cmin:confidence?0:0,cmax:confidence?100:320,showscale:!isSample,colorbar:{{title:confidence?'Model SD<br>(g/kg)':'Mean SOC<br>(g/kg)',thickness:16}},line:{{color:isSample?'#8e1d18':'#ffffff',width:isSample?1.8:.4}},symbol:isSample?'diamond':'circle'}}}};
}}
function update(){{
 const rows=selected(),t=+thresholdEl.value,metric=metricEl.value;
 thresholdValue.textContent=t.toFixed(2)+' g/kg';
 const sample=rows.filter(d=>sendTeam(d,t)),trust=rows.filter(d=>!sendTeam(d,t));
 trustCount.textContent=trust.length.toLocaleString();sampleCount.textContent=sample.length.toLocaleString();
 const pct=100*sample.length/rows.length; sampleArea.textContent=pct.toFixed(1)+'% of represented tile area needs sampling';
 lowKpi.textContent=pct.toFixed(1)+'%';lowKpiSub.textContent=regionEl.value+' • threshold '+t.toFixed(2)+' g/kg';
 const b=bounds[regionEl.value];
 Plotly.react('map',[trace(trust,'Trust',metric,t,false),trace(sample,'Send team',metric,t,true)],{{
  margin:{{l:60,r:30,t:48,b:55}},paper_bgcolor:'#fff',plot_bgcolor:'#f7f9fa',hovermode:'closest',
  title:{{text:(metric==='confidence'?'MODEL AGREEMENT / SAMPLING NEED':'ENSEMBLE MEAN SOC PREDICTION')+' — '+regionEl.options[regionEl.selectedIndex].text, x:.5,font:{{size:15,color:'#17365d'}}}},
  xaxis:{{title:'Longitude',range:b.x,gridcolor:'#dce3e9',zeroline:false}},yaxis:{{title:'Latitude',range:b.y,gridcolor:'#dce3e9',zeroline:false}},
  legend:{{orientation:'h',x:.01,y:1.03}},uirevision:regionEl.value
 }},{{responsive:true,displaylogo:false}});
 const ranked=[...sample].sort((a,b)=>(Number(b.any_seed_negative)-Number(a.any_seed_negative)) || b.model_std_gkg-a.model_std_gkg).slice(0,+topNEl.value);
 worklist.innerHTML=ranked.map((d,i)=>`<tr class="${{d.any_seed_negative?'impossible':''}}"><td class="rank">${{i+1}}</td><td>${{d.sample_index}}</td><td>${{d.latitude.toFixed(4)}}°</td><td>${{d.longitude.toFixed(4)}}°</td><td>${{d.model_std_gkg.toFixed(2)}} g/kg</td><td>${{d.any_seed_negative?'<span class="badge">NEGATIVE</span>':'LOW'}}</td></tr>`).join('') || '<tr><td colspan="6">No tiles currently require sampling.</td></tr>';
}}
[regionEl,metricEl,thresholdEl,topNEl].forEach(el=>el.addEventListener('input',update)); update();
</script></body></html>"""


def build_notebook(html: str) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# SoilTrust - District Soil Intelligence\n\n"
            "**The soil map that tells you where to trust it - and where to send ground samples.**\n\n"
            "This notebook embeds the complete offline dashboard below. The data are fixed outputs from "
            "five previously trained continental-transfer models; this notebook performs no training."
        ),
        nbf.v4.new_code_cell(
            "# Rebuild from the source CSV and summary when needed:\n"
            "# %run build_dashboard.py\n"
            "# The executed output below is embedded and does not require a notebook server.\n"
            "from IPython.display import HTML\n"
            "HTML(open('soiltrust_dashboard.html', encoding='utf-8').read())",
            execution_count=1,
            outputs=[nbf.v4.new_output("display_data", data={"text/html": html}, metadata={})],
        ),
    ]
    nbf.write(nb, NOTEBOOK_PATH)


def main() -> None:
    records, summary = load_data()
    html = build_html(records, summary)
    HTML_PATH.write_text(html, encoding="utf-8")
    build_notebook(html)
    print(json.dumps({
        "rows": len(records), "regions": {r: sum(x["region"] == r for x in records) for r in ["North America", "Europe"]},
        "html": str(HTML_PATH), "html_bytes": HTML_PATH.stat().st_size,
        "notebook": str(NOTEBOOK_PATH), "notebook_bytes": NOTEBOOK_PATH.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
