class RuleEngine:
    def __init__(self):
        pass

    def apply_rules(self, transaction):
        try:
            # ✅ Convert safely
            amount = float(transaction.get('transaction_amount', 0))
            browser = int(transaction.get('payer_browser_anonymous', -1))
            mode = int(transaction.get('transaction_payment_mode_anonymous', -1))
            mobile = transaction.get('payer_mobile_anonymous', '')

            # Rule 1: High amount via mobile
            if amount > 10000 and transaction.get('transaction_channel') == 'mobile':
                return True, "high_amount_mobile"

            # Rule 2: Weird browser and low payment mode
            if browser == 12 and mode == 0:
                return True, "weird_browser_low_mode"

            # Rule 3: Missing mobile number
            if mobile.lower() == 'unknown':
                return True, "missing_mobile_number"

        except Exception as e:
            print(f"Rule Engine Error: {e}")

        return False, None
