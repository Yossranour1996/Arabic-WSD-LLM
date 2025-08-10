#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
parse-wsd-log.py  –  Build a prediction file that mirrors the Dataset-A
                     WSD test-truth format and optionally evaluate it.

Usage:
    python prediction.py 
         --log debug_info.txt 
         --truth test_truth.json 
         --pred predictions.json 
         --eval  eval_report.json          # ← eval is optional
"""

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ----------------------------------------------------------------------
# 0. Regex that captures *one* Sense ID from the model’s answer
# ----------------------------------------------------------------------
#
# Put your Swiss-army pattern here.  **It must contain exactly one
# capturing group** (the digits).  Below is a *minimal* example that
# matches things like
#
#     The correct Sense ID is **14704**
#     I would choose 14704
#
# You will certainly replace / expand it.

SENSE_REGEX = re.compile(r"""
    (?ixs)  # i = ignore case, x = allow verbose regex, s = dot matches newline

# i = case-insensitive, x = verbose, s = dot-all
    (?:                       # ── any ONE of the templates below ───────────

        # 1 I would choose  Sense ID: N
        .*? i \s+ would \s+ choose \b .*? sense \s+ id \s* :? \s* \** \s* (\d+) \s* \**

      | # 2 I choose Sense ID: N
        .*? i \s+ choose \b        .*? sense \s+ id \s* :? \s* \** \s* (\d+) \s* \**

      | # 3 Correct Sense ID: N
        .*? correct \s+ sense \s+ id \s* : \s* \** \s* (\d+) \s* \**

      | # 4 Correct Sense ID is: N
        .*? correct \s+ sense \s+ id \s+ is \s* :? \s* \** \s* (\d+) \s* \**

      | # 5 I would choose the correct sense as: N
        .*? i \s+ would \s+ choose \s+ the \s+ correct \s+ sense \s+ as \s* :? \s* \** \s* (\d+) \s* \**

      | # 6 The correct answer is: N
        .*? the \s+ correct \s+ answer \s+ is \s* :? \s* \** \s* (\d+) \s* \**

      | # 7 My answer is …  **Sense ID: N**
        .*? my \s+ answer \s+ is \b .*? sense \s+ id \s* :? \s* \** \s* (\d+) \s* \**

      | # 8 I would select Sense ID: N
        .*? i \s+ would \s+ select \b .*? sense \s+ id \s* :? \s* \** \s* (\d+) \s* \**
        
    
      # | # 9  correct Sense ID for … is  <N>   (N may be bare, or **N**, or “Sense ID: N”)
      #   .*? correct \s+ sense \s+ id \s+ for \b .*? \s+ is \s* :? \s*
      #       (?: \** \s* (?: sense \s+ id \s* :? \s* )? )?   # optional '**' and/or 'Sense ID:'
      #       \** \s* (\d+) \s* \**
            

| # 9b – The correct sense ID for ... is Sense ID: N (comma or period allowed after number)
  .*? correct \s+ sense \s+ id \s+ for \b .*? \s+ is \s* :? \s*
      (?: sense \s+ id \s* :? \s* )?   # now this comes AFTER 'is'
      \**? \s* (\d+) \s* \**? \s* [.,]?

       | # 10  correct Sense ID is indeed N   (N or **N**)
        correct \s+ sense \s+ id \s+ is \s+ indeed \s* :? \s* \** (\d+) \**

       | # 11  I would choose the sense with ID: N   (N or **N**)
        .*? i \s+ would \s+ choose \s+ the \s+ sense \s+ with \s+ id \s* :? \s* \** (\d+) \**
        
       | # 12 I would choose: [optional newline] then number in any format
        .*? i \s+ would \s+ choose \s* : \s* (?: \n \s* )?    # "I would choose:" with optional newline
        \** \s*                                          # optional opening **
        (?: sense \s+ id \s* :? \s* )?                   # optional "Sense ID:" or "Sense ID"
        (\d+)\s*[.,]?                                           # ← the number we want to capture
        \s* \**  
        
        | #13 I would select Sense ID: N (even with leading text and optional **, colons, etc.)
        .*?                              # allow leading context like “Therefore,”
        i \s+ would \s+ select \s+       
        \**? \s*                        
        sense \s+ id                   
        \s* :? \s*                     
        \**? \s* (\d+) \s* \**?        
       # the number with optional **
                                # optional closing **
      | # 14  I believe that Sense ID N (or with **, or colon)
        .*? i \s+ believe \s+ that \s+
        \**? \s*
        (?:sense \s+ id \s* :? \s*)?     # optional "Sense ID:" or "Sense ID"
        \**? \s* (\d+) \s* \**?         # the number (optionally wrapped)
        \s+ is \s+ the \s+ correct \s+ (?:sense|choice|answer)
        
        | # 15 I choose sense ID N (with optional colon or **)
        i \s+ choose \s+ sense \s+ id \s* :? \s* \**? \s* (\d+) \s* \**?
       
       |# 16 I would choose: (then newline, then Sense ID … N in any form)
        .*? i \s+ would \s+ choose \s* : \s* \n+
        \s* \** \s*
        (?: sense \s+ id \s* :? \s* )?
        (\d+)
        \s* \**
        
      | # 17  "The correct sense for ... is ... [any text] ... Sense ID: N"
        .*? the \s+ correct \s+ sense \s+ for \b .*? \s+ is \b .*?
            (?:sense\s+id\s*:?\s*)?   # Optional "Sense ID:" (any case)
            \**?\s*(\d+)\s*\**?\s*[.,]?  # Capture N (with optional **, comma/period)

      | # 18  "The correct sense is ... [any text] ... Sense ID: N"
        .*? the \s+ correct \s+ sense \s+ is \b .*?
            (?:sense\s+id\s*:?\s*)?   # Optional "Sense ID:" (any case)
            \**?\s*(\d+)\s*\**?\s*[.,]?  # Capture N
| #18  in the given sentence is N  (N, **N**, or “Sense ID N”)
  .*? in \s+ the \s+ given \s+ sentence \s+ is \s* :? \s*
      (?: \**? \s* (?: sense \s+ id \s* :? \s* )? )?   # optional ** and/or “Sense ID”
      \**? \s* (\d+) \s* \**?                          # ← capture the number

| #19  in this sentence is N  (allows a line break and “Sense ID:”)
  .*? in \s+ this \s+ sentence \s+ is \s* :? \s*
      (?: \n \s* )?                                    # optional blank line
      \**? \s* (?: sense \s+ id \s* :? \s* )?          # optional “Sense ID:”
      \**? \s* (\d+) \s* \**?
      
| #18  in the given sentence is N  (N, **N**, or “- Sense ID: N”)
  .*? in \s+ the \s+ given \s+ sentence \s+ is \s* :? \s*
      (?: \n \s* )?                                     # optional newline
      -? \s* \**? \s*                                   # optional dash + **
      (?: sense \s+ id \s* :? \s* )?                    # optional "Sense ID:"
      \**? \s* (\d+) \s* \**?

| #19  in this sentence is N  (same as above)
  .*? in \s+ this \s+ sentence \s+ is \s* :? \s*
      (?: \n \s* )?                                     # optional newline
      -? \s* \**? \s*
      (?: sense \s+ id \s* :? \s* )?
      \**? \s* (\d+) \s* \**?

| #20  appears to be - Sense ID: N (e.g., at the end of sentence)
  .*? appears \s+ to \s+ be \s* :? \s*
      -? \s* \**? \s*                                   # optional dash + **
      (?: sense \s+ id \s* :? \s* )?
      \**? \s* (\d+) \s* \**?
| #21  correct sense … is: \n Sense ID: N
  .*? correct \s+ sense .*? is \s* : \s*        # "… is:"
  \n+ \s*                                       # newline after colon
  -? \s* \**? \s*                               # optional dash + **
  (?: sense \s+ id \s* : \s* )?                 # optional "Sense ID:"
  \**? \s* (\d+) \s* \**? ,?
  | #22  I believe the correct Sense ID is: N / **N**
  .*? i \s+ believe \s+ the \s+ correct \s+ sense \s+ id \s+ is \s* :? \s*
  \**? \s* (\d+) \s* \**?
| #23  Therefore, the correct Sense ID is: N / **N**
  .*? therefore ,? \s+ the \s+ correct \s+ sense \s+ id \s+ is \s* :? \s*
  \**? \s* (\d+) \s* \**?
 #24 
|  (?m: ^\s* (\d+)\s*[.,]?         # number, optional , or .
        \s* (?: Definition \b .* )? # optional “Definition: …”
        \s* $ )
#25
|  (?mx:                            # m = multiline, x = verbose
      ^\s* -? \s* \**? \s*          # optional dash and **
      Sense \s+ ID \s* : \s*
      \**? \s* (\d+)\s*[.,]? \s*    # ← capture N, allow , or .
      $ )
    # 23  Bare number on its own line, e.g. “394” or “394,”
|    (?m) ^\s* (\d+) \s* [.,]? \s*$

  | # 24  Stand-alone bullet “- Sense ID: N” (comma/period optional)
    -? \s* \**? \s*
    Sense \s+ ID \s* :? \s*
    \**? \s* (\d+) \s* \**? \s* [.,]?

  | # 25  “in this context is … [Sense ID:] N”
    .*? in \s+ this \s+ context \s+ is \s* :? \s*
        (?: \n \s* )?                   # optional newline after colon
        -? \s* \**? \s*
        (?: sense \s+ id \s* :? \s* )?  # optional “Sense ID:”
        \**? \s* (\d+) \s* \**?

  | # 26  “the correct sense from the list is … [Sense ID:] N”
    .*? the \s+ correct \s+ sense \s+ from \s+ the \s+ list \s+ is \s* :? \s*
        (?: \n \s* )?                   # optional newline
        -? \s* \**? \s*
        (?: sense \s+ id \s* :? \s* )?  # optional “Sense ID:”
        \**? \s* (\d+) \s* \**?
   | # 27  “the correct Sense ID is likely …”  (optional colon, **, period)
    .*? the \s+ correct \s+ sense \s+ id \s+ is \s+ likely \s* :? \s*
        \**? \s* (\d+) \s* \**? \.?   
        
     # capture N, allow bold or trailing period
  |    (?m: ^\s* (\d+) \s* [.,]? .* $)
    )""")

# ----------------------------------------------------------------------
# 1. Parse the .txt log into blocks
# ----------------------------------------------------------------------

# helper patterns (no flags because they are anchored at ^)
HDR_QUERY = re.compile(r"^===== Query for Word:\s*(.*?)\s*=====")
LINE_SENT  = re.compile(r'^Sentence:\s*"(.*)"')
LINE_RESP  = re.compile(r"^Response:")

def parse_log(path: Path) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts:
        {
          'sentence':   <arabic sentence string>,
          'word':       <target word>,
          'answer_raw': <entire answer block as one string>
        }
    """
    blocks: List[Dict[str, Any]] = []
    cur = None
    collecting = False
    buf: List[str] = []

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            ln = raw.rstrip("\n")

            # ---- header ---------------------------------------------
            m = HDR_QUERY.match(ln)
            if m:
                # flush previous
                if cur:
                    cur["answer_raw"] = "\n".join(buf)
                    blocks.append(cur)
                cur = {"word": m.group(1), "sentence": None}
                buf = []
                collecting = False
                continue

            # ---- sentence line --------------------------------------
            if not collecting:
                ms = LINE_SENT.match(ln)
                if ms and cur:
                    cur["sentence"] = ms.group(1).strip()
                # not yet in answer block, skip
                if LINE_RESP.match(ln):
                    collecting = True
                    tail = ln.split("Response:", 1)[1].strip()
                    if tail:
                        buf.append(tail)
                continue

            # ---- collect answer lines -------------------------------
            if collecting:
                buf.append(ln)

    # flush last block
    if cur:
        cur["answer_raw"] = "\n".join(buf)
        blocks.append(cur)

    return blocks

# ----------------------------------------------------------------------
# 2. Extract exactly one sense ID from an answer
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 2. Extract one (and only one) Sense-ID from an answer block
# ----------------------------------------------------------------------
def extract_sense(text: str) -> str:
    """
    Scan `text` (the part between the *Response:* line and the next header).

    • If the regex finds one and only one distinct number → return it.
    • If it finds zero or >1 different numbers            → return "none".
    """
    matches = SENSE_REGEX.findall(text)        # list of tuples
    # Flatten the tuples (because each alternative in the big pattern
    # has one capturing group) and keep non-empty strings only.
    numbers = [m for tup in matches for m in tup if m]

    uniq = set(numbers)
    if len(uniq) == 1:
        return uniq.pop()          # exactly one unique Sense-ID
    return "none"                  # 0 or ≥2 different IDs  → ambiguous


# ----------------------------------------------------------------------
# 3. Build a prediction dict that mirrors the gold structure
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 3. Build a prediction dict that mirrors the gold structure
#       **sequentially, block-by-block**
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 3. Build predictions strictly *block-by-block*
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 3. Build predictions – tolerate missing / extra blocks
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 3. Build predictions – tolerate gaps *and* maintain alignment
# ----------------------------------------------------------------------
def build_predictions(blocks, gold):
    """
    Aligns predictions to gold words by order *and* surface check.

    • If the header word of the next block == the next gold word → use it.
    • Otherwise treat the gold word as missing a prediction → "none".
    • Continues until all gold words are processed.
    • Prints lists of words that got "none" and blocks that were ignored.
    """
    pred           = deepcopy(gold)
    missing_report = []           # (sentence_id, word_id, word)
    extra_blocks   = []           # blocks left over at the end

    blk_idx = 0
    num_blocks = len(blocks)

    for sent in pred:
        sid = sent["sentence_id"]
        for w in sent["words"]:
            gold_word = w["word"].strip()

            if blk_idx < num_blocks and blocks[blk_idx]["word"].strip() == gold_word:
                # Perfect alignment – consume the block
                w["target_sense"] = extract_sense(blocks[blk_idx]["answer_raw"])
                blk_idx += 1
            else:
                # No matching block -> "none"
                w["target_sense"] = "none"
                missing_report.append((sid, w["word_id"], gold_word))

    # Any blocks we didn’t consume are “extra”
    if blk_idx < num_blocks:
        extra_blocks = blocks[blk_idx:]

    # -------  console summaries  --------------------------------------
    if missing_report:
        print(f"[INFO] {len(missing_report)} gold words got 'none' "
              "(no matching prediction):")
        for sid, wid, word in missing_report:
            print(f"   sentence_id={sid:<6} word_id={wid:<6} word='{word}'")

    if extra_blocks:
        print(f"[INFO] {len(extra_blocks)} prediction blocks were ignored "
              "(no matching gold word). "
              f"First ignored header word: '{extra_blocks[0]['word']}'")

    return pred


# ----------------------------------------------------------------------
# 4. Optional evaluation
# ----------------------------------------------------------------------

def evaluate(pred: List[Dict[str, Any]],
             gold: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Writes y_true / y_pred by (sentence_id, word_id) alignment.
    """
    y_true, y_pred = [], []

    gold_index: Dict[Tuple[int, int], str] = {}
    for s in gold:
        sid = s["sentence_id"]
        for w in s["words"]:
            gold_index[(sid, w["word_id"])] = str(w["target_sense"])

    for s in pred:
        sid = s["sentence_id"]
        for w in s["words"]:
            key = (sid, w["word_id"])
            y_pred.append(str(w["target_sense"]))
            y_true.append(gold_index[key])

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(y_true, y_pred) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    return {
        "Accuracy (%)": round(acc, 2),
        "Precision":    round(prec, 4),
        "Recall":       round(rec, 4),
        "F1":           round(f1, 4),
        "Instances":    len(y_true),
    }

# ----------------------------------------------------------------------
# 5. CLI
# ----------------------------------------------------------------------

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log",   required=True, type=Path)
    ap.add_argument("--truth", required=True, type=Path)
    ap.add_argument("--pred",  required=True, type=Path,
                    help="where to write the prediction JSON")
    ap.add_argument("--eval",  type=Path,
                    help="if set, run evaluation and dump the report here")
    args = ap.parse_args()

    blocks = parse_log(args.log)

    gold = json.loads(args.truth.read_text(encoding="utf-8"))
    predictions = build_predictions(blocks, gold)

    args.pred.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ predictions written to {args.pred}")

    if args.eval:
        report = evaluate(predictions, gold)
        args.eval.write_text(
            json.dumps(report, indent=4, ensure_ascii=False), encoding="utf-8"
        )
        print("📊 evaluation:")
        print(json.dumps(report, indent=4, ensure_ascii=False))
        print(f"✅ eval report written to {args.eval}")

# ----------------------------------------------------------------------

if __name__ == "__main__":
    cli()
