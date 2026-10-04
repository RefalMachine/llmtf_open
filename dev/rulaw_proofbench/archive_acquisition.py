"""One-time archive assembly from the documented curl acquisition cache.

Normal reproduction uses the committed archive and generate.py/validate.py.
This script records the exact primary URLs and bytes, never reconstructs laws
from model knowledge. The optional cache path contains the downloaded files.
"""
import argparse
import json
from pathlib import Path

try:
    from .generate import ROOT, digest
    from .source_tools import extract, article
except ImportError:
    from generate import ROOT, digest
    from source_tools import extract, article

AMENDMENTS={'2017-139':'102437322','2017-125':'102435732','2013-185':'102166739',
    '2020-127':'102722391','2008-49':'102121402','2012-302':'102162486','2013-100':'102165201',
    '2014-357':'102362473','2013-182':'102166365','2018-528':'102500789','2023-480':'605772018',
    '2024-498':'608103570','2024-405':'607920359','2024-547':'608103494'}
CLAUSES={'tk':[92,93,94,128,173,177,287], 'sk':[17,19,21,57,81,83,99],
         'gk':[21,26,27,28,37,186,188], '59':[1,8,11,12]}


def assemble(cache):
    support=[]
    def add(ident,kind,raw,url):
        text=extract(raw)
        ext='mht' if raw.startswith(b'MIME-Version:') else 'html'
        a='sources/'+ident+'.'+ext;b='sources/'+ident+'.txt'
        (ROOT/a).write_bytes(raw);(ROOT/b).write_text(text,encoding='utf-8')
        support.append(dict(id=ident,kind=kind,url=url,raw_path=a,text_path=b,
                            sha256=digest(raw),text_sha256=digest(text.encode()),retrieved_at='2026-10-04'))
        return text
    for ident,nd in AMENDMENTS.items():
        add('amendment-'+ident,'amendment_original',(cache/('rulaw-amend-'+ident+'.mht')).read_bytes(),
            f'http://pravo.gov.ru/proxy/ips/?savertf=&nd={nd}&rdk=0')
    manifest=json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
    for s in manifest:
        k=s['id'];rdk={'tk':'182','sk':'53','gk':'141','59':'10'}[k]
        text=add(k+'-comparison','later_consolidated_comparison',(cache/('rulaw-'+k+'-next.mht')).read_bytes(),
                 f'http://pravo.gov.ru/proxy/ips/?savertf=&nd={s["nd"]}&rdk={rdk}')
        original=(ROOT/s['text_path']).read_text(encoding='utf-8')
        equality={str(n):article(original,n)==article(text,n) for n in CLAUSES[k]}
        if not all(equality.values()):raise ValueError(f'Changed selected fragment {k}: {equality}')
        s.update(edition_audit_status='verified_selected_fragments',comparison_source_id=k+'-comparison',
                 unchanged_articles=equality,
                 verification_interval={'from':'2025-01-01','through':'2025-01-01',
                    'meaning':'Point-in-time audit of selected fragments; not a whole-code edition validity interval.'},
                 edition_audit_file='edition_audit.json')
        if s['raw_path'].endswith('.mht'):
            s['url']=f'http://pravo.gov.ru/proxy/ips/?savertf=&nd={s["nd"]}&rdk={s["rdk"]}'
        for suffix in ('metadata','editions'):
            path=ROOT/'sources'/(k+'-'+suffix+'.html')
            url=f'http://pravo.gov.ru/proxy/ips/?'+('doc_itself=&vkart=card' if suffix=='metadata' else 'docbody=')+f'&nd={s["nd"]}'+(f'&rdk={s["rdk"]}' if suffix=='metadata' else '&page=all')
            add(k+'-'+suffix,'portal_metadata',path.read_bytes(),url)
    (ROOT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'supporting_sources.json').write_text(json.dumps(support,ensure_ascii=False,indent=2)+'\n')
    print(f'Archived {len(manifest)} main and {len(support)} supporting documents')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,default=Path('/tmp'))
    assemble(p.parse_args().cache)
