---
title: HS Code Classifier
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# AI-Powered HS Code Classifier POC

This Proof of Concept (POC) is a bilingual (Arabic/English) Streamlit application that classifies products into Harmonized System (HS) Codes.

It accepts either:
1. A **Product Description** (in English or Arabic).
2. A **GTIN Barcode** (8 to 14 digits). If a GTIN is provided, the app uses the OpenFoodFacts API to look up the product name before classification.

## Deploying to Hugging Face Spaces

This repository is configured to be deployed directly to Hugging Face Spaces using the Streamlit SDK.
