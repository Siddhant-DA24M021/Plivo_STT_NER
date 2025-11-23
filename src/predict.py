import json
import argparse
import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, LABEL2ID, label_is_pii
import os


def normalize_email(text: str):
    t = text.lower()
    t = t.replace(" dot ", ".")
    t = t.replace(" underscore ", "_")
    t = t.replace(" at ", "@")
    return t

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def is_valid_email(span: str):
    span = normalize_email(span)
    return EMAIL_RE.search(span) is not None

def is_valid_phone(span: str):
    digits = "".join([c for c in span if c.isdigit()])
    return len(digits) >= 7

def is_valid_credit_card(span: str):
    digits = "".join([c for c in span if c.isdigit()])
    return 13 <= len(digits) <= 19

def is_valid_date(span: str):
    span = span.lower()
    has_digits = any(ch.isdigit() for ch in span)
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul",
              "aug", "sep", "oct", "nov", "dec"]
    has_month = any(m in span for m in months)
    return has_digits or has_month



def fix_bio_sequence(pred_ids):
    labels = [ID2LABEL[i] for i in pred_ids]
    cleaned = []
    prev_ent = None
    prev_tag = "O"

    for lab in labels:
        if lab == "O":
            cleaned.append("O")
            prev_ent, prev_tag = None, "O"
            continue

        tag, ent = lab.split("-", 1)

        if tag == "B":
            cleaned.append(lab)
            prev_tag, prev_ent = "B", ent

        else:  # I-tag
            if prev_tag in ("B", "I") and prev_ent == ent:
                cleaned.append(lab)
                prev_tag, prev_ent = "I", ent
            else:
                cleaned.append("O")
                prev_tag, prev_ent = "O", None

    return cleaned


##########################################
# ---------- BIO → Span Conversion -------
##########################################
def bio_to_spans(text, offsets, labels):
    spans = []
    current_label = None
    start_idx = None
    end_idx = None

    for (start, end), lab in zip(offsets, labels):
        if start == 0 and end == 0:
            continue

        if lab == "O":
            if current_label is not None:
                spans.append((start_idx, end_idx, current_label))
                current_label = None
            continue

        prefix, ent = lab.split("-", 1)

        if prefix == "B":
            if current_label is not None:
                spans.append((start_idx, end_idx, current_label))
            current_label = ent
            start_idx = start
            end_idx = end
        else:  # I
            if current_label == ent:
                end_idx = end
            else:
                if current_label is not None:
                    spans.append((start_idx, end_idx, current_label))
                current_label = ent
                start_idx = start
                end_idx = end

    if current_label is not None:
        spans.append((start_idx, end_idx, current_label))

    return spans


##########################################
# --------------- MAIN -------------------
##########################################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name
    )
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    CONF_THRESH = 0.60  # good trade-off

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )

            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                probs = torch.softmax(logits, dim=-1)
                max_probs, pred_raw = probs.max(dim=-1)

            # Apply confidence filtering
            pred_ids = []
            for i, pid in enumerate(pred_raw.tolist()):
                if ID2LABEL[pid] != "O" and max_probs[i] < CONF_THRESH:
                    pred_ids.append(LABEL2ID["O"])
                else:
                    pred_ids.append(pid)

            # BIO cleanup
            cleaned_labels = fix_bio_sequence(pred_ids)

            # Convert BIO → spans
            spans = bio_to_spans(text, offsets, cleaned_labels)

            # Validate PII spans
            ents = []
            for s, e, lab in spans:
                span_text = text[s:e]

                # Validation per type
                if lab == "EMAIL" and not is_valid_email(span_text):
                    continue
                if lab == "PHONE" and not is_valid_phone(span_text):
                    continue
                if lab == "CREDIT_CARD" and not is_valid_credit_card(span_text):
                    continue
                if lab == "DATE" and not is_valid_date(span_text):
                    continue

                ents.append({
                    "start": int(s),
                    "end": int(e),
                    "label": lab,
                    "pii": bool(label_is_pii(lab)),
                })

            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
