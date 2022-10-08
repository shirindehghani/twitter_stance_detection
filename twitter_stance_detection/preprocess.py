import re

def preprocess(col):
    out_col = []
    for text in col.values.tolist():
        text = text.replace("\n", " ")
        text = text.replace("\s+", " ")
        text = re.sub(r"(.)\1+",r"\1", text)
        new_text=[]
        for t in text.split(" "):
            t = "@user" if t.startswith('@') and len(t) > 1 else t
            t = "http" if t.startswith('htp') else t
            new_text.append(t)
        new_text = " ".join(new_text)
        out_col.append(new_text)
    return out_col