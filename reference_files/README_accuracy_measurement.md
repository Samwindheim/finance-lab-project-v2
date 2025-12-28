# Accuracy Measurement of Investor datapoints

For this step, we want to extract the investors/underwriters/guarantors from *all* the PDF sources in the database, and compare them to the manually extracted datapoints already stored in the database.

The goal is to measure the accuracy of the extraction process, and identify any common failure modes that can be improved. It is also possible that the manual extraction itself contains errors, which we may identify and correct during this process.

The output of this step should be a report (markdown or similar) that includes the following information:
- Total number of documents processed.
- Total number of investors/underwriters/guarantors extracted.
- Number of correct datapoint extractions (matching the manual data).
- Number of incorrect datapoint extractions (not matching the manual data).
- Number of missing datapoint extractions (present in manual data but not extracted).
- Number of false positives (datapoints extracted but not present in manual data).
- Number of documents with 100% accuracy.
- Number of documents with less than 100% accuracy.

For each incorrect or missing datapoint, include details such as:
- Document name.
- Extracted datapoint (if any).
- Manual datapoint.
- Type of error (incorrect value, missing value, false positive).
- Possible reason for the error (if identifiable).

The output should initially be a json file with all the relevant data, and then a summarized report in markdown format.

## Input files

- `sources.json`: A json file containing a list of links for all PDF sources in the database, with some metadata.
- `manual_investors.json`: A json file containing the extracted investors/underwriters/guarantors for each entity (issue/warrant/convertible).

## Output files

- `accuracy_report.json`: A json file containing the detailed accuracy report for each document.
- `accuracy_summary.md`: A markdown file summarizing the accuracy metrics and findings.

## Steps to perform

1. Load the `sources.json` file.
2. For each PDF source, download the PDF file and use the existing extraction process to extract the investors/underwriters/guarantors.
3. Store the extracted datapoints in a structured format (e.g., json) for easy access.
4. Load the `manual_investors.json` file.
5. For each document, compare the extracted datapoints to the manual datapoints.
6. Record the accuracy metrics and details of any discrepancies.
7. Generate the `accuracy_report.json` file with all the detailed data.
8. Summarize the findings and metrics in the `accuracy_summary.md` file.

I suggest implementing the steps in such a way that they can be called separately, to allow for easier debugging and validation of each step. Store intermediate results in separate files as needed.

Another thing to note:
If both percentage and monetary values are listed in the sources, we only record the monetary values (and calculate percentages in the app). Percentages are only stored when no monetary value is provided. I think you should still extract both when possible, but when measuring accuracy, bear this in mind.
In some cases, just names are listed, no percentages nor monetary values.
Convertibles don't have a "level".