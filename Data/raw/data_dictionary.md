# Data Dictionary

## Original Data Source: Transportation Management System (TMS)

### Core Fields
| Field Name | Data Type | Description |
|------------|-----------|-------------|
| `shipment_id` | string | Unique identifier for each shipment |
| `ship_date` | datetime | Date when shipment was picked up |
| `lane` | string | Origin → Destination route |
| `invoice_amount` | float | Total amount charged ($) |
| `billable_weight` | float | Weight used for billing (kg) |
| `base_charge` | float | Core freight cost ($) |
| `fuel_surcharge` | float | Fuel-related additional charge ($) |
| `service_level` | string | Shipping service type |

### Derived Fields (Created During Analysis)
| Field Name | Description | Business Logic |
|------------|-------------|----------------|
| `cost_per_kg` | Invoice Amount / Billable Weight | Standard efficiency metric |
| `needs_review` | Data quality flag | TRUE for anomalies, service fees with minimal weight |
| `cleaned_billable_weight` | Corrected weight for analysis | 0 for service fees, actual weight for freight |
| `charge_type` | Cost categorization | Standard Freight, Fixed Fee Service, Minimum Charge |