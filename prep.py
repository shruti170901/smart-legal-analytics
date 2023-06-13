from __future__ import unicode_literals, print_function
import os
import re
import sys
import spacy
nlp = spacy.load("en_core_web_trf")
from spacy.lang.en import English
from spacy.attrs import ORTH


def show(count, j):
    x = int(60*j/count)
    print("[{}{}] {}/{}".format("-"*x, " "*(60-x), j, count), 
        end='\r', file=sys.stdout, flush=True)

#%% MAIN
data_source_folder = 'judis_2000'
data_target_folder = 'judis_150_spacy_ORTH'

data_files = os.listdir(data_source_folder)
iterator_data = 0

for fn in data_files: 
        iterator_data += 1
        file_old_contents = open(data_source_folder+'/'+fn,'r',encoding='utf8')
        file_new_contents = file_old_contents.read()
        file_old_contents.close()

        file_new_contents = file_new_contents.splitlines()
        file_new_contents = ' '.join(file_new_contents)

        file_new_contents = re.sub(r'\s+', ' ', file_new_contents, 0, flags=re.I)
        file_new_contents = re.sub(r'http://JUDIS.NIC.IN SUPREME COURT OF INDIA Page [0-9]+ of [0-9]+', '', file_new_contents, 0, flags=re.I)
        file_new_contents = re.sub(r'([\.\!\?\-\;\:\*\'\"\&\\\/])+', r'\1', file_new_contents)

        nlp = English()
        nlp.add_pipe('sentencizer')

        special_cases = {
            "Rs.": "Rs.", "No.": "No.", "Nos.":"Nos", "NO.": "No.", "no.": "No.", "vs.": "vs", "i.e.": "i.e.", "viz.": "viz.", "M/s.": "M/s.", "Mohd.": "Mohd.", "Ex.": "exhibit", 
            "Art." : "article", "Arts." : "articles", "S.": "section", "Sec.": "section", "s.": "section", "ss.": "sections", "u/s.": "section", "u/ss.": "sections", "art.": "article",
            "arts.": "articles", "u/arts." : "articles", "u/art." : "article", "Sch.": "Sch", "Art.": "Art", "sub-cl.": "sub-cl", "cl.": "cl", "Mad.": "Mad", 
            "M/s.": "M/s", "S.C.R.": "S.C.R", "Exs.": "Exs", "Messrs.": "Messrs", "CO.":"CO", "I.A.": "I.A", "I.T.R.":"I.T.R", "Ltd.":"Ltd", "I.T.C.":"I.T.C", 
            "I.L.R.":"I.L.R", "a.":"a", "M.L.J.":"M.L.J", "A.I.R.":"A.I.R", "R.A.A.":"R.A.A", "A.C.":"A.C", "Lah.":"Lah", "Bom.":"Bom", "Cal.":"Cal", "A.T.R.":"A.T.R", 
            "Co.":"Co", "Bros.":"Bros", "Rang.":"Rang", "Ltd.":"Ltd", "Q.B.":"Q.B", "L.R.":"L.R", "I.A.":"I.A", "I.M.R.": "I.M.R", "para.":"para", "G.O.":"G.O", 
            "Dr.":"Dr", "All.":"All", "C.W.N.":"C.W.N", "P.C.":"P.C", "S.L.P.":"S.L.P", "W. P.":"W.P", "W.P.":"W.P", "P.":"P", "Govt.":"Govt", "Const.":"Const", 
            "P.M.":"P.M", "S. C.":"S.C", "S.C.":"S.C", "S.L.P.":"S.L.P", "S. L. P.":"S.L.P", "N.V.K.":"N.V.K", "F.I.R.":"F.I.R", "P.W.":"P.W", "P. W.":"P.W",
            "P.Ws.":"P.Ws", "A.M.":"A.M", "Mst.":"Mst", "Smt.":"Smt", "S.C.C.":"S.C.C", "sub-sec.":"sub-sec", "Addl.":"Addl", "Anr.":"Anr", "sub-s.":"sub-s", 
            "S.R.":"S.R", "O.As.":"O.As", "C.A.":"C.A", "Ors.":"Ors", "Supp.":"Supp", "O.M.":"O.M", "T.P.":"T.P", "Vs.":"Vs", "LTD.":"LTD", "SCh.":"SCh", "Lodg.":"Lodg", "Vol.":"Vol", "(":"(",
            "ORS.":"ORS", "ETC.":"ETC", "sq.":"sq", "mts.":"mts", "Supl.":"Supl", "ANR.":"ANR", "MR.":"MR", "etc.":"etc", "Sri.":"Sri", "Shri.":"Shri", "Spl.":"Spl", "Adv.":"Adv",
            "Sr.":"Sr", "Fed.":"Fed", "Pvt.":"Pvt", "Advs.":"Advs", "DR.":"DR", "Dr.":"Dr", "SMT.": "SMT", "LRS.":"LRS", "ft.":"ft", "Cr.":"Cr", "w.e.f.":"w.e.f",
            "M.Sc.":"M.Sc", "B.Sc.":"B.Sc"
         }
         # more at http://www.commonlii.org/in/journals/NLUDLRS/2010/11.pdf
             
        for case, orth in special_cases.items():
            nlp.tokenizer.add_special_case(case, [{ORTH: case}])

        doc = nlp(file_new_contents)
        file_new_contents = [sent.text.strip() for sent in doc.sents]
        file_new_contents = '\n'.join(file_new_contents)

        file_new_contents = re.sub(r'([\.\?\!\"\'\.])\s*\(\s*',r'\1 (',file_new_contents)

        f_updated = open(data_target_folder+'/'+fn,'w',encoding='utf8')
        f_updated.write(file_new_contents)
        f_updated.close()
        
        show(data_files.__len__(),iterator_data)