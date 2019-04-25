# Cyber-data-analytics_TUDelft

DATA DESCRIPTION:

- issuercountrycode:  
  country where the card was issued
  
- txvariantcode:  
  the card type that was used (subbrand of visa or master card)
  
- bin:  
  card issuer identifier
  
- amount/currencycode:  
  transaction amount in minor units (so 100 EUR = 100 euro cent)
  
- shoppercountrycode:  
  IP address country
  
- shopperinteraction:  
  Ecommerce if it was an online transaction, ContAuth if it was a (monthly) subscription
  
- simple_journal:  
  Payment status.  
  Settled = “transaction approved and no fraud reported”  
  Refused = “transaction was declined, can be fraud, but can also be insufficient funds, etc”  
  Chargeback = “transaction was approved, but turned out to be fraud”  
  
- bookingdate:  
  only relevant for Chargebacks.  
  Time stamp when the chargeback was reported. During simulation you may only use this knowledge after this date.  
  So for example if on an email address a transaction ended in a chargeback, you can only block that email address after the booking date of the chargeback.  

- cardverificationresponsesupplied:  
  did the shopper provide his 3 digit CVC/CVV2 code?  
  
- cvcresponsecode:  
  Validation result of the CVC/CVV2 code:  
  0 = Unknown,  
  1 = Match,  
  2 = No Match,  
  3-6 = Not checked  
  
- creationdate:  
  Date of transaction  
  
- accountcode:  
  merchant’s webshop  
  
- mail_id:  
  Email address  
  
- ip_id:  
  Ip address  
  
- card_id:  
  Card number  
  
