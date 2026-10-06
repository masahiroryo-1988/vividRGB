
from research_paths import research_path
import urllib.request,json
from pathlib import Path
dois=['10.1093/icesjms/fsz171','10.1038/s43705-021-00085-1','10.1007/BF02289233','10.1016/0377-0427(87)90125-7','10.1214/aos/1176344136','10.1038/s41598-021-95616-0']
out={}
for doi in dois:
    try:
        req=urllib.request.Request('https://api.crossref.org/works/'+doi,headers={'User-Agent':'FFA-draft-reference-check'})
        z=json.load(urllib.request.urlopen(req,timeout=25))['message']
        out[doi]={k:z.get(k) for k in ['title','author','container-title','published','volume','issue','page','article-number','URL']}
    except Exception as e:out[doi]={'error':str(e)}
p=Path(str(research_path('work/ffa-preprint/data/reference_metadata.json')));p.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
