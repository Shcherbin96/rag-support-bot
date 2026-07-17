# Per-case Evaluation Results

| Model | Case | Type | OK | Route | Sources | Error |
|---|---|---|---:|---|---|---|
| google/gemini-2.5-flash-lite | shipping_price | grounded | True | factual_in_domain | 02_shipping.md | — |
| google/gemini-2.5-flash-lite | free_shipping | grounded | True | factual_in_domain | 01_about.md, 02_shipping.md | — |
| google/gemini-2.5-flash-lite | return_period | grounded | True | factual_in_domain | 04_returns.md | — |
| google/gemini-2.5-flash-lite | payment_methods | grounded | False | factual_in_domain | — | model_contract_error |
| google/gemini-2.5-flash-lite | cash_payment | grounded | True | factual_in_domain | 03_payment.md | — |
| google/gemini-2.5-flash-lite | installments | grounded | True | factual_in_domain | 03_payment.md | — |
| google/gemini-2.5-flash-lite | receipt | grounded | True | factual_in_domain | 03_payment.md | — |
| google/gemini-2.5-flash-lite | warranty_appliances | grounded | True | factual_in_domain | 05_warranty.md, 10_catalog.md | — |
| google/gemini-2.5-flash-lite | rewards_rate | grounded | True | factual_in_domain | 07_rewards.md | — |
| google/gemini-2.5-flash-lite | welcome_promo | grounded | True | factual_in_domain | 06_promotions.md | — |
| google/gemini-2.5-flash-lite | support_hours | grounded | True | factual_in_domain | 08_contact.md | — |
| google/gemini-2.5-flash-lite | support_phone | grounded | True | factual_in_domain | 08_contact.md | — |
| google/gemini-2.5-flash-lite | support_contact | grounded | True | factual_in_domain | 08_contact.md | — |
| google/gemini-2.5-flash-lite | business_orders | grounded | False | factual_in_domain | 03_payment.md | — |
| google/gemini-2.5-flash-lite | gift_order | grounded | False | factual_in_domain | — | model_contract_error |
| google/gemini-2.5-flash-lite | change_order | grounded | True | factual_in_domain | 09_faq.md | — |
| google/gemini-2.5-flash-lite | international_shipping | grounded | True | factual_in_domain | 02_shipping.md, 09_faq.md | — |
| google/gemini-2.5-flash-lite | catalog_kettles | grounded | True | factual_in_domain | 01_about.md, 10_catalog.md | — |
| google/gemini-2.5-flash-lite | mixed_greeting_shipping | grounded | True | factual_in_domain | 02_shipping.md | — |
| google/gemini-2.5-flash-lite | unknown_warehouse | refuse | True | out_of_domain | — | — |
| google/gemini-2.5-flash-lite | unknown_revenue | refuse | False | factual_in_domain | — | model_contract_error |
| google/gemini-2.5-flash-lite | unrelated_tesla | refuse | True | out_of_domain | — | — |
| google/gemini-2.5-flash-lite | unrelated_weather | refuse | True | out_of_domain | — | — |
| google/gemini-2.5-flash-lite | unrelated_rain | refuse | True | out_of_domain | — | — |
| google/gemini-2.5-flash-lite | unrelated_cardiology | refuse | True | out_of_domain | — | — |
| google/gemini-2.5-flash-lite | unrelated_border | refuse | True | out_of_domain | — | — |
| google/gemini-2.5-flash-lite | unrelated_other_store | refuse | True | out_of_domain | — | — |
| google/gemini-2.5-flash-lite | unrelated_movie | refuse | True | out_of_domain | — | — |
| google/gemini-2.5-flash-lite | adversarial_prompt | refuse | True | adversarial | — | — |
| google/gemini-2.5-flash-lite | adversarial_mixed | refuse | True | adversarial | — | — |
| google/gemini-2.5-flash-lite | greeting_en | smalltalk | True | smalltalk | — | — |
| google/gemini-2.5-flash-lite | thanks_en | smalltalk | True | smalltalk | — | — |
| google/gemini-2.5-flash-lite | greeting_hi | smalltalk | True | smalltalk | — | — |
