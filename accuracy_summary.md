# Extraction Accuracy Summary

## Overall Metrics
- **Total Documents Processed:** 10
- **Total Investors Extracted (Predicted):** 93
- **Correct Datapoint Extractions:** 65
- **Incorrect Datapoint Extractions:** 15
- **Missing Datapoint Extractions:** 55
- **False Positives:** 13
- **Documents with 100% Accuracy:** 2
- **Documents with < 100% Accuracy:** 8

## Discrepancy Details

### 📄 ITAB_Prospekt.pdf
- **Type:** False Positive
  - **Extracted:** `Erik Selin Fastigheter AB`
- **Type:** Incorrect Value
  - **Investor:** `Svolder AB`
  - **Errors:** Amount mismatch: Did not extract an amount where one was expected (43774508.000)
- **Type:** Incorrect Value
  - **Investor:** `Stig-Olof Simonsson`
  - **Errors:** Amount mismatch: Did not extract an amount where one was expected (31293652.000)
- **Type:** False Positive
  - **Extracted:** `Investeraren`
- **Type:** Incorrect Value
  - **Investor:** `Andréas Elgaard`
  - **Errors:** Level mismatch: Got 0, expected 1
- **Type:** Missing
  - **Expected:** `Anders Moberg`
- **Type:** Missing
  - **Expected:** `?`
- **Type:** Missing
  - **Expected:** `ISTMO AB`
- **Type:** Missing
  - **Expected:** `Övre Kullen AB`
- **Type:** Missing
  - **Expected:** `Andréas Elgaard`
- **Type:** Missing
  - **Expected:** `SOMACHE AB`
- **Type:** Missing
  - **Expected:** `Aeternum Capital AS`
- **Type:** Missing
  - **Expected:** `Fredrik Rapp`
- **Type:** Missing
  - **Expected:** `Fredrik Rapp`
- **Type:** Missing
  - **Expected:** `Petter Fägersten`
- **Type:** Missing
  - **Expected:** `VIEM Invest AB`

### 📄 GTG_2023-11-08_Memorandum.pdf
- **Type:** False Positive
  - **Extracted:** `Esseff Fastigheter Aktiebolag`
- **Type:** Missing
  - **Expected:** `Esseff Fastigheter AB`

### 📄 PURE_2022‑01‑27_Prospekt.pdf
- **Type:** Missing
  - **Expected:** `Money Never Sleeps Holding AB`
- **Type:** Missing
  - **Expected:** `?`

### 📄 ODIN_2025‑04‑01_Memorandum.pdf
- **Type:** Missing
  - **Expected:** `Consentia Group AB`
- **Type:** Missing
  - **Expected:** `Martin Jonsson`
- **Type:** Missing
  - **Expected:** `Fields of Gold Holding AB`
- **Type:** Missing
  - **Expected:** `Martin Olauson`
- **Type:** Missing
  - **Expected:** `Christer Hager`
- **Type:** Missing
  - **Expected:** `Seventh Sense Adventures Holding AB`
- **Type:** Missing
  - **Expected:** `Marcus Andersson`
- **Type:** Missing
  - **Expected:** `Nestero Holding`
- **Type:** Missing
  - **Expected:** `Björn Wallin`
- **Type:** Missing
  - **Expected:** `Per Nellgård`
- **Type:** Missing
  - **Expected:** `Red one and red two AB`
- **Type:** Missing
  - **Expected:** `Pronator Invest AB`
- **Type:** Missing
  - **Expected:** `Paginera Invest AB`
- **Type:** Missing
  - **Expected:** `Andreas Poike`
- **Type:** Missing
  - **Expected:** `Anders Haskel`
- **Type:** Missing
  - **Expected:** `Nils Berg`
- **Type:** Missing
  - **Expected:** `Meriti Neutral`
- **Type:** Missing
  - **Expected:** `John Andersson Moll`
- **Type:** Missing
  - **Expected:** `Karkas Capital`
- **Type:** Missing
  - **Expected:** `Jens Miöen`
- **Type:** Missing
  - **Expected:** `UBB Consulting AB`
- **Type:** Missing
  - **Expected:** `Aktia Bank Plc`
- **Type:** Missing
  - **Expected:** `Anders Husmark`
- **Type:** Missing
  - **Expected:** `Viktor Westman`

### 📄 GUARD_2021‑10‑20_Prospekt.pdf
- **Type:** Missing
  - **Expected:** `?`
- **Type:** Missing
  - **Expected:** `?`
- **Type:** Missing
  - **Expected:** `M2 Asset Management AB`
- **Type:** Missing
  - **Expected:** `M2 Asset Management AB`
- **Type:** Missing
  - **Expected:** `Jan Ståhlberg`
- **Type:** Missing
  - **Expected:** `Jan Ståhlberg`

### 📄 TESSIN_2024-06-14_Memorandum.pdf
- **Type:** Missing
  - **Expected:** `Dan Brander`

### 📄 REDW_2023-08-25_Prospekt.pdf
- **Type:** Incorrect Value
  - **Investor:** `Johan Waldhe`
  - **Errors:** Amount mismatch: Got 2599999.000, expected 2599999.800
- **Type:** Incorrect Value
  - **Investor:** `Sutjagin Capital AB`
  - **Errors:** Amount mismatch: Got 1749999.000, expected 1749999.900
- **Type:** Incorrect Value
  - **Investor:** `Arne Andersson`
  - **Errors:** Amount mismatch: Got 999999.000, expected 999999.900
- **Type:** Incorrect Value
  - **Investor:** `Consentia Group AB`
  - **Errors:** Amount mismatch: Got 999999.000, expected 999999.900
- **Type:** Incorrect Value
  - **Investor:** `AD94 Holding AB`
  - **Errors:** Amount mismatch: Got 799999.000, expected 799999.800
- **Type:** Incorrect Value
  - **Investor:** `Pronator Invest AB`
  - **Errors:** Amount mismatch: Got 699999.000, expected 699999.900
- **Type:** Incorrect Value
  - **Investor:** `Elvil AB`
  - **Errors:** Amount mismatch: Got 499999.000, expected 499999.800
- **Type:** Incorrect Value
  - **Investor:** `Mattias Wachtmeister`
  - **Errors:** Amount mismatch: Got 368524.000, expected 368524.800
- **Type:** Incorrect Value
  - **Investor:** `Erik Svensson`
  - **Errors:** Amount mismatch: Got 349999.000, expected 349999.800
- **Type:** Incorrect Value
  - **Investor:** `Pierre Heneen`
  - **Errors:** Amount mismatch: Got 199999.000, expected 199999.800
- **Type:** Incorrect Value
  - **Investor:** `Rickard Danielsson`
  - **Errors:** Amount mismatch: Got 199999.000, expected 199999.800
- **Type:** Incorrect Value
  - **Investor:** `Arne Andersson`
  - **Errors:** Amount mismatch: Got 645126.000, expected 645126.600

### 📄 IMSYS_2022-06-21_Prospekt.pdf
- **Type:** False Positive
  - **Extracted:** `Magnus Stuart, VD och styrelseledamot`
- **Type:** False Positive
  - **Extracted:** `Stefan Blixt, CTO, advisor och styrelseledamot`
- **Type:** False Positive
  - **Extracted:** `Henry Sténson, Styrelseordförande`
- **Type:** False Positive
  - **Extracted:** `Stefan Mårtensson, aktieägare`
- **Type:** False Positive
  - **Extracted:** `Anders Malmqvist, aktieägare`
- **Type:** False Positive
  - **Extracted:** `Samosa Consulting AB, aktieägare`
- **Type:** False Positive
  - **Extracted:** `Anders Gradén, aktieägare`
- **Type:** False Positive
  - **Extracted:** `Marcus Risland, f.d styrelseledamot`
- **Type:** False Positive
  - **Extracted:** `Frank Schubert, styrelseledamot`
- **Type:** False Positive
  - **Extracted:** `Jan Wäreby, styrelseledamot`
- **Type:** Missing
  - **Expected:** `Stefan Mårtensson`
- **Type:** Missing
  - **Expected:** `Stefan Blixt`
- **Type:** Missing
  - **Expected:** `Samosa Consulting AB`
- **Type:** Missing
  - **Expected:** `Magnus Stuart`
- **Type:** Missing
  - **Expected:** `Marcus Risland`
- **Type:** Missing
  - **Expected:** `Jan Wäreby`
- **Type:** Missing
  - **Expected:** `Frank Schubert`
- **Type:** Missing
  - **Expected:** `Anders Gradén`
- **Type:** Missing
  - **Expected:** `Anders Malmqvist`
- **Type:** Missing
  - **Expected:** `Henry Stenson`
