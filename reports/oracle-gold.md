# Eval report for oracle

- ts: `2026-07-19T15:23:26+00:00`
- mode: `algorithm (oracle)`
- label: `oracle`
- data: `fixtures/gold.jsonl`
- convention: `conventions/default.yaml`
- examples: 100
- validity: 100.0%
- exact record: 100.0%
- field accuracy: 100.0% (1323/1323)

## Peer table (fill base / fine-tuned when scored)

| System         | Validity | Field accuracy |
| -------------- | -------- | -------------- |
| oracle         | 100.0%   | 100.0%         |
| base SLM       |          |                |
| fine-tuned SLM |          |                |

## Per-category field accuracy

_No category labels in this fixture._

## Per-field support (gold fields compared)

| Field      | Support | Hits | Accuracy |
| ---------- | ------- | ---- | -------- |
| amount     | 26      | 26   | 100.0%   |
| baseUnit   | 26      | 26   | 100.0%   |
| city       | 64      | 64   | 100.0%   |
| country    | 64      | 64   | 100.0%   |
| currency   | 100     | 100  | 100.0%   |
| email      | 64      | 64   | 100.0%   |
| houseNo    | 64      | 64   | 100.0%   |
| iban       | 64      | 64   | 100.0%   |
| legalForm  | 64      | 64   | 100.0%   |
| name1      | 100     | 100  | 100.0%   |
| name2      | 31      | 31   | 100.0%   |
| phone      | 64      | 64   | 100.0%   |
| postalCode | 64      | 64   | 100.0%   |
| recordId   | 100     | 100  | 100.0%   |
| recordType | 100     | 100  | 100.0%   |
| status     | 100     | 100  | 100.0%   |
| street     | 64      | 64   | 100.0%   |
| validFrom  | 100     | 100  | 100.0%   |
| vatId      | 64      | 64   | 100.0%   |
