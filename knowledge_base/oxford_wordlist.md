# OXFORD 3000 WORDLIST (CEFR-STRUCTURED)

## OVERVIEW
Oxford 3000 adalah daftar 3000 kata paling penting dalam bahasa Inggris.
Digunakan sebagai dasar:
- Vocabulary learning path
- Error correction
- AI feedback system

Level coverage:
A1 → B2

---

# FORMAT DATA

Setiap entry:
- word
- pos (part of speech)
- level
- optional: tag

---

# A1 VOCABULARY

## CORE WORDS

### Function Words
- a / an (article)
- the (article)
- and (conj)
- but (conj)
- or (conj)
- because (conj)
- in (prep)
- on (prep)
- at (prep)
- to (prep)

### Pronouns
- I (pron)
- you (pron)
- he (pron)
- she (pron)
- it (pron)
- we (pron)
- they (pron)

### Verbs (Basic)
- be (v)
- have (v)
- do (v)
- go (v)
- get (v)
- make (v)
- take (v)

### Daily Objects
- house (n)
- car (n)
- book (n)
- phone (n)
- food (n)
- water (n)

### People & Social
- man (n)
- woman (n)
- friend (n)
- family (n)
- teacher (n)

### Descriptive
- good (adj)
- bad (adj)
- big (adj)
- small (adj)
- new (adj)
- old (adj)

---

# A2 VOCABULARY

## CORE EXTENSION

### Verbs
- accept (v)
- achieve (v)
- allow (v)
- answer (v)
- arrive (v)
- believe (v)

### Abstract Nouns
- ability (n)
- accident (n)
- advantage (n)
- behaviour (n)
- experience (n)

### Adjectives
- active (adj)
- careful (adj)
- comfortable (adj)
- difficult (adj)
- important (adj)

### Communication
- communicate (v)
- discussion (n)
- explain (v)
- message (n)

---

# B1 VOCABULARY

## INTERMEDIATE USAGE

### Verbs
- analyse (v)
- argue (v)
- compare (v)
- decide (v)
- improve (v)

### Abstract Concepts
- development (n)
- education (n)
- environment (n)
- knowledge (n)

### Adjectives
- advanced (adj)
- independent (adj)
- responsible (adj)

---

# B2 VOCABULARY

## ADVANCED CORE

### Verbs
- acknowledge (v)
- assume (v)
- confirm (v)
- establish (v)
- evaluate (v)

### Academic Words
- analysis (n)
- concept (n)
- context (n)
- method (n)
- theory (n)

### Complex Adjectives
- appropriate (adj)
- significant (adj)
- relevant (adj)

---

# TAGGING SYSTEM (UNTUK AI)

## CATEGORY TAG
- daily
- academic
- business
- social

## DIFFICULTY TAG
- high-frequency
- medium
- advanced

---

# USAGE FOR YOUR SYSTEM

## 1. Vocabulary Scoring
Bandingkan output user dengan wordlist:
- % A1 words
- % A2 words
- % B1+ words

## 2. Feedback AI
Contoh:
- "You are using mostly A1 vocabulary"
- "Try using more B1-level words like: improve, compare"

## 3. Error Detection
Mapping:
- wrong word → suggestion from same level

## 4. Learning Recommendation
- Recommend next-level vocabulary

---

# EXAMPLE OUTPUT (AI)

Input:
- user speech

Output:
- Vocabulary Level: A2
- Weak: lack of B1 vocabulary
- Suggestion:
  - replace "very good" → "excellent"
  - replace "big" → "large"

---

# CHUNKING STRATEGY (PENTING)

Split berdasarkan:
- level (A1, A2, B1, B2)
- category (verbs, nouns, etc.)

Contoh chunk:
"A2 verbs list"
"B1 academic vocabulary"

---

# NOTE (IMPORTANT)

- Jangan masukkan semua 3000 kata dalam 1 chunk
- Ideal: 100–200 words per chunk
- Simpan sebagai embedding-friendly text

---