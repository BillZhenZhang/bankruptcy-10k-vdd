#!/usr/bin/env python3
import argparse, csv, hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parent; AUDIT=ROOT/"reliability_audit"

def hit(text,pattern): return re.search(pattern,text,re.I) is not None

def classify(row):
    clean=lambda text: re.sub(r"\s+", " ", re.sub(r"\[/?target\]", "", text.lower())).strip()
    s=clean(row["sentence_text_masked"]); c=clean(row["adjacent_context_masked"]+" "+row["sentence_text_masked"]); p=row["phrase"]
    generic=hit(c,r"\b(asu|fasb|accounting standards? update|authoritative guidance|pronouncement|subtopic 205-40)\b") or hit(s,r"\b(customary|usual and customary|includes? events? of default|contains? events? of default|provides? for events? of default|events? of default include)\b")
    compliant=hit(c,r"\b(in compliance with (?:all|the)|no event of default|not in default|do not anticipate (?:a )?(?:covenant )?violation|does not suggest .*not be able to meet obligations)\b")
    modal=hit(s,r"\b(if|unless|could|would|might|potential(?:ly)?|prospective|should circumstances|in the event)\b") or hit(s,r"\bmay (?:be|have|need|not|result|cause|trigger|seek|require|constitute|lead|become|experience|choose|determine|occur|remain|fail)\b")
    resolved=hit(c,r"\b(waived|waiver (?:was )?(?:obtained|received|granted)|cured|remedied|resolved|repaid|no longer)\b")
    if p=="hard_going_concern":
        if generic or hit(c,r"\b(requires? management to (?:perform|evaluate|assess)|management must specifically disclose|effective for (?:annual|fiscal)|early adoption)\b"): return "SUPPRESSED","going_concern_standard"
        if hit(c,r"\b(no|not) substantial doubt\b") or hit(c,r"\bsubstantial doubt (?:has been|was) alleviated\b"): return "SUPPRESSED","going_concern_negated"
        if hit(c,r"\b(there is substantial doubt|management (?:has )?concluded .*substantial doubt|auditor|independent registered public accounting firm|opinion .*expressing substantial doubt|explanatory paragraph|emphasis paragraph)\b"): return "RISK_ACTIVE","going_concern_current"
        if hit(s,r"\b(our|company'?s) ability to continue as a going concern\b") and hit(c,r"\b(dependent|depends|uncertain|outside of our control|may not|could)\b"): return "RISK_ACTIVE","going_concern_prospective"
        return "SUPPRESSED","going_concern_other"
    if p=="hard_unable_obligations":
        if compliant or hit(c,r"\binsurers? require the collateral\b"): return "SUPPRESSED","unable_negated_or_insurance"
        if hit(s,r"\b(if|may|could|would).*unable|unable.*\b(may|could|would)\b"): return "RISK_ACTIVE","unable_prospective"
        return "SUPPRESSED","unable_other"
    if p=="hard_forbearance_waiver":
        if hit(s,r"\bforbearance agreement\b") and hit(c,r"\b(entered into|our lenders|with the bank|current portion|remains in effect|extended the standstill|lenders? .*forbear|delay .*remedies|payment blockage|specified defaults?)\b"): return "RISK_ACTIVE","active_forbearance"
        if hit(s,r"\b(?:have )?not received .*waiver from our lenders\b") or hit(c,r"\b(?:were|are) not in compliance\b") and hit(s,r"\b(?:obtained|received) .*waiver from our lenders\b"): return "RISK_ACTIVE","waiver_linked_to_current_noncompliance"
        if modal and hit(s,r"\b(waiver|forbearance)\b") and (hit(c,r"\b(no assurances|unable to obtain|absent a waiver)\b") or not hit(c,r"\b(expect to remain|were in compliance with all)\b")): return "RISK_ACTIVE","prospective_waiver"
        if resolved or compliant: return "SUPPRESSED","waiver_received_or_compliant"
        return "SUPPRESSED","waiver_other"
    if p=="hard_default_acceleration":
        if compliant: return "SUPPRESSED","default_negated"
        if generic: return "SUPPRESSED","default_boilerplate"
        if not hit(s,r"^\s*if\b|\bprohibits? .* if\b|\bcould occur if\b") and hit(s,r"\b(?:we|company|borrower) (?:are|is|were|was|have been|has been) (?:currently )?in default\b"): return "RISK_ACTIVE","current_default"
        return "SUPPRESSED","default_other"
    if p=="hard_covenant_noncompliance":
        if not hit(s,r"^\s*if\b") and hit(s,r"\b(?:are|is|were|was) (?:not in compliance|in violation|in breach)\b"): return "RISK_ACTIVE","current_covenant"
        if compliant or hit(c,r"\b(do not anticipate|did not result in|would not constitute)\b"): return "SUPPRESSED","covenant_negated"
        if generic: return "SUPPRESSED","covenant_boilerplate"
        if resolved and not hit(c,r"\b(?:currently|still|continuing) (?:not in compliance|in violation|in breach)\b"): return "SUPPRESSED","covenant_resolved"
        if hit(s,r"\b(?:without causing|could result in|would represent|would,? if .* result in|should .* be breached|unable to obtain .* financial .*breach)\b"): return "RISK_ACTIVE","prospective_covenant"
        return "SUPPRESSED","covenant_other"
    return "SUPPRESSED","excluded_family"

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--evaluate-heldout",action="store_true"); args=parser.parse_args()
    with (AUDIT/"channel1_three_reviewer_consensus.csv").open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    split="held_out_rule_validation" if args.evaluate_heldout else "rule_development"; out=[]
    for row in rows:
        gold=row["consensus_label"] or "AM"
        if row["phrase"]=="hard_bankruptcy_restructuring" or row["annotation_split"]!=split: continue
        prediction,rule=classify(row); truth="RISK_ACTIVE" if gold in {"RA","PR"} else "SUPPRESSED"
        out.append({**row,"gold_label":gold,"truth_binary":truth,"predicted_binary":prediction,"rule_id":rule,"correct":int(truth==prediction)})
    tp=sum(r["truth_binary"]==r["predicted_binary"]=="RISK_ACTIVE" for r in out); fp=sum(r["truth_binary"]=="SUPPRESSED" and r["predicted_binary"]=="RISK_ACTIVE" for r in out); fn=sum(r["truth_binary"]=="RISK_ACTIVE" and r["predicted_binary"]=="SUPPRESSED" for r in out); tn=len(out)-tp-fp-fn
    summary={"split":split,"rows":len(out),"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    suffix="heldout" if args.evaluate_heldout else "development"; fields=list(out[0])
    with (AUDIT/f"context_rule_{suffix}_predictions.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(out)
    (AUDIT/f"context_rule_{suffix}_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
