# Extraction Accuracy Summary

## Overall Metrics
- **Total Documents Processed:** 43
- **Total Investors Extracted (Predicted):** 815
- **Correct Datapoint Extractions:** 762
- **Incorrect Datapoint Extractions:** 0
- **Missing Datapoint Extractions:** 9
- **False Positives:** 53
- **Documents with 100% Accuracy:** 38
- **Documents with < 100% Accuracy:** 5

## Discrepancy Details

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

  Reason: Query not grabbing right page, missed page 5

### 📄 TESSIN_2024-06-14_Memorandum.pdf
- **Type:** Missing
  - **Expected:** `Dan Brander`

  Reason: Query not grabbing right page, konvertible, missed page 2

### 📄 DMYD-B_2023-06-02_Supplement.pdf
- **Type:** False Positive
  - **Extracted:** `Formue Nord Markedsneutral A/S`
- **Type:** False Positive
  - **Extracted:** `Selandia Alpha Invest A/S`
- **Type:** False Positive
  - **Extracted:** `Patrik Hansen`
- **Type:** False Positive
  - **Extracted:** `Bernhard Osten Sacken`
- **Type:** False Positive
  - **Extracted:** `Capono AB`
- **Type:** False Positive
  - **Extracted:** `Nowo Global Fund`
- **Type:** False Positive
  - **Extracted:** `Carl Rosvall`
- **Type:** False Positive
  - **Extracted:** `Martin Bjäringer`
- **Type:** False Positive
  - **Extracted:** `Buntel AB`
- **Type:** False Positive
  - **Extracted:** `Nolsterby Invest AB`
- **Type:** False Positive
  - **Extracted:** `Lusam Invest AB`
- **Type:** False Positive
  - **Extracted:** `Wilhelm Risberg`
- **Type:** False Positive
  - **Extracted:** `Arne Grundström`
- **Type:** False Positive
  - **Extracted:** `G&W Holding AB`
- **Type:** False Positive
  - **Extracted:** `Excelity AB`
- **Type:** False Positive
  - **Extracted:** `QQM Hedge Master`
- **Type:** False Positive
  - **Extracted:** `Gryningskust Holding AB`
- **Type:** False Positive
  - **Extracted:** `Spikarna Förvaltning AB`
- **Type:** False Positive
  - **Extracted:** `Oliver Molse`
- **Type:** False Positive
  - **Extracted:** `Oscar Molse`
- **Type:** False Positive
  - **Extracted:** `Anders Johansson`
- **Type:** False Positive
  - **Extracted:** `ATH Invest AB`
- **Type:** False Positive
  - **Extracted:** `Accrelium AB`
- **Type:** False Positive
  - **Extracted:** `Bearpeak AB`
- **Type:** False Positive
  - **Extracted:** `Pegroco Holding AB`
- **Type:** False Positive
  - **Extracted:** `Visa Invest AB`
- **Type:** False Positive
  - **Extracted:** `Pronator Invest AB`
- **Type:** False Positive
  - **Extracted:** `Strategic Wisdom Nordic AB`
- **Type:** False Positive
  - **Extracted:** `Haldran AB`
- **Type:** False Positive
  - **Extracted:** `Consentia Group AB`
- **Type:** False Positive
  - **Extracted:** `Cordouan Invest AB`
- **Type:** False Positive
  - **Extracted:** `Ghanen Chouha`
- **Type:** False Positive
  - **Extracted:** `Hans Wernstedt`
- **Type:** False Positive
  - **Extracted:** `Hälsa Invest Sweden AB`
- **Type:** False Positive
  - **Extracted:** `Johan Waldhe`
- **Type:** False Positive
  - **Extracted:** `Lycksele Telemarketing AB`
- **Type:** False Positive
  - **Extracted:** `Mats Carlsson`
- **Type:** False Positive
  - **Extracted:** `MIHAB AB`
- **Type:** False Positive
  - **Extracted:** `Tony Chouha`
- **Type:** False Positive
  - **Extracted:** `XC Invest AB`
- **Type:** False Positive
  - **Extracted:** `Erik Lundin`
- **Type:** False Positive
  - **Extracted:** `Jan Robert Pärsson`
- **Type:** False Positive
  - **Extracted:** `Maria Eldrot Lundmark`
- **Type:** False Positive
  - **Extracted:** `P&M CF AB`
- **Type:** False Positive
  - **Extracted:** `Peter Lundmark`
- **Type:** False Positive
  - **Extracted:** `Rune Löderup`
- **Type:** False Positive
  - **Extracted:** `Max Björs`

  Reason: Crossed out / cancelled guarantors, extraction has trouble detecting that so it still grabbed them

### 📄 TOBII_2024-03-18_Prospekt.pdf
- **Type:** False Positive
  - **Extracted:** `Jörgen Lantto`
- **Type:** False Positive
  - **Extracted:** `Mats Backman`
- **Type:** False Positive
  - **Extracted:** `Ann Emilson`
- **Type:** False Positive
  - **Extracted:** `Emma Bauer`
- **Type:** False Positive
  - **Extracted:** `Jonas Jakstad`
- **Type:** False Positive
  - **Extracted:** `Patrick Grundler`

  Reason: Manual doesn't include some garantors, they have 0%

### 📄 SVED-B_2024-01-04_Prospekt.pdf
- **Type:** Missing
  - **Expected:** `Nordea Fonder`
- **Type:** Missing
  - **Expected:** `If Skadeförsäkring`

  Reason: Not grabbing right page, "Nyckelinformation om emittenten"