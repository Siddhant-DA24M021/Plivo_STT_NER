import json
import random
import os
from datetime import datetime, timedelta

# ------------------------------------------------
# Your STT helper pools
# ------------------------------------------------

names = [
    "ramesh", "suresh", "john", "rohan", "priyanka", "pooja",
    "arjun", "kavita", "megha", "rahul", "vignesh", "santosh"
]

domains = ["gmail dot com", "yahoo dot com", "outlook dot com", "hotmail dot com"]

cities = ["chennai", "mumbai", "delhi", "kolkata", "hyderabad", "pune", "jaipur"]
locations = ["bus stand", "railway station", "airport", "city center", "main market"]

digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


# ------------------------------------------------
# Helper functions to generate entities
# ------------------------------------------------

def random_phone_stt():
    return " ".join(random.choice(digits) for _ in range(10))

def random_phone_numeric():
    return "".join(str(random.randint(0, 9)) for _ in range(10))

def random_email():
    name = random.choice(names)
    lname = random.choice(names)
    dom = random.choice(domains)
    return f"{name} dot {lname} at {dom}"

def random_credit_card():
    nums = [str(random.randint(0, 9)) for _ in range(16)]
    return " ".join("".join(nums[i:i+4]) for i in range(0, 16, 4))

def random_date():
    start = datetime(2021, 1, 1)
    d = start + timedelta(days=random.randint(0, 1400))
    return d.strftime("%d %B %Y").lower()


# ------------------------------------------------
# Templates
# ------------------------------------------------

templates = [
    "my email is {EMAIL}",
    "my credit card number is {CREDIT_CARD}",
    "call me on {PHONE}",
    "i will travel on {DATE}",
    "i live in {CITY}",
    "i am currently at the {LOCATION}",
    "reach me at {EMAIL} and my phone is {PHONE}",
    "my number is {PHONE} and email is {EMAIL}",
    "please contact {PERSON_NAME} on {PHONE}",
    "send details to {EMAIL} before {DATE}",
]


# ------------------------------------------------
# Build a labeled or unlabeled sample
# ------------------------------------------------

def build_sample(idx, labeled=True):
    t = random.choice(templates)
    text = t
    entities = []

    # EMAIL
    if "{EMAIL}" in text:
        e = random_email()
        s = text.index("{EMAIL}")
        text = text.replace("{EMAIL}", e)
        entities.append({"start": s, "end": s+len(e), "label": "EMAIL"})

    # CREDIT CARD
    if "{CREDIT_CARD}" in text:
        cc = random_credit_card()
        s = text.index("{CREDIT_CARD}")
        text = text.replace("{CREDIT_CARD}", cc)
        entities.append({"start": s, "end": s+len(cc), "label": "CREDIT_CARD"})

    # PHONE
    if "{PHONE}" in text:
        p = random.choice([random_phone_numeric(), random_phone_stt()])
        s = text.index("{PHONE}")
        text = text.replace("{PHONE}", p)
        entities.append({"start": s, "end": s+len(p), "label": "PHONE"})

    # DATE
    if "{DATE}" in text:
        d = random_date()
        s = text.index("{DATE}")
        text = text.replace("{DATE}", d)
        entities.append({"start": s, "end": s+len(d), "label": "DATE"})

    # CITY
    if "{CITY}" in text:
        c = random.choice(cities)
        s = text.index("{CITY}")
        text = text.replace("{CITY}", c)
        entities.append({"start": s, "end": s+len(c), "label": "CITY"})

    # LOCATION
    if "{LOCATION}" in text:
        l = random.choice(locations)
        s = text.index("{LOCATION}")
        text = text.replace("{LOCATION}", l)
        entities.append({"start": s, "end": s+len(l), "label": "LOCATION"})

    # PERSON NAME (extra)
    if "{PERSON_NAME}" in text:
        pname = random.choice(names) + " " + random.choice(names)
        s = text.index("{PERSON_NAME}")
        text = text.replace("{PERSON_NAME}", pname)
        entities.append({"start": s, "end": s+len(pname), "label": "PERSON_NAME"})

    if labeled:
        return {"id": f"utt_{idx:04d}", "text": text, "entities": entities}
    return {"id": f"utt_{idx:04d}", "text": text}


# ------------------------------------------------
# Append new samples to existing files
# ------------------------------------------------

def append_jsonl(path, items):
    with open(path, "a", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x) + "\n")


if __name__ == "__main__":

    # You decide how many new samples to add
    NEW_TRAIN = 500
    NEW_DEV = 100
    NEW_TEST = 100

    # Generate new ones
    train_new = [build_sample(i+3000, labeled=True) for i in range(NEW_TRAIN)]
    dev_new = [build_sample(i+4000, labeled=True) for i in range(NEW_DEV)]
    test_new = [build_sample(i+5000, labeled=False) for i in range(NEW_TEST)]

    # Append to existing files
    append_jsonl("data/train.jsonl", train_new)
    append_jsonl("data/dev.jsonl", dev_new)
    append_jsonl("data/test.jsonl", test_new)

    print("Bro, new data appended successfully!")
