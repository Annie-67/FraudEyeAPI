from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback
from rule_engine import RuleEngine
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from flask_cors import CORS

# ------------------ DATABASE SETUP ------------------
engine = create_engine('sqlite:///fraud.db', echo=False)
Base = declarative_base()

# New table for storing input transaction details
class TransactionDetails(Base):
    __tablename__ = 'transaction_details'
    id = Column(Integer, primary_key=True)
    transaction_id = Column(String)
    transaction_data = Column(JSON)  # Storing transaction data as JSON

# Renamed FraudTransaction table to FraudPredictions
class FraudPrediction(Base):
    __tablename__ = 'fraud_prediction'
    id = Column(Integer, primary_key=True)
    transaction_id = Column(String)
    is_fraud = Column(Boolean)
    fraud_source = Column(String)
    fraud_reason = Column(String)
    fraud_score = Column(Float)

# Renamed ReportedFraud table to FraudReports
class FraudReports(Base):
    __tablename__ = 'fraud_reports'
    id = Column(Integer, primary_key=True)
    transaction_id = Column(String)
    reported_by_user = Column(Boolean)
    manual_fraud_label = Column(Integer)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# ------------------ APP INIT ------------------
model = joblib.load("stacking_fraud_model.pkl")
rule_engine = joblib.load("rule_engine.pkl")

app = Flask(__name__)
CORS(app)

# ------------------ ENCODING MAPS ------------------
channel_map = {'web': 0, 'mobile': 1}
email_map = {'normaluser@example.com': 0}
mobile_map = {'XXXXX111.0': 0}
transaction_id_map = {'ANON_99': 0}
payee_map = {'ANON_0': 0}

# ------------------ PREPROCESS FUNCTION ------------------
def preprocess(txn):
    df_input = pd.DataFrame([txn])
    df_input.drop(columns=['transaction_date'], errors='ignore', inplace=True)

    df_input['transaction_channel'] = df_input['transaction_channel'].map(channel_map)
    df_input['payer_email_anonymous'] = df_input['payer_email_anonymous'].map(email_map)
    df_input['payer_mobile_anonymous'] = df_input['payer_mobile_anonymous'].map(mobile_map)
    df_input['transaction_id_anonymous'] = df_input['transaction_id_anonymous'].map(transaction_id_map)
    df_input['payee_id_anonymous'] = df_input['payee_id_anonymous'].map(payee_map)

    # ✅ Convert everything to numeric
    df_input = df_input.apply(pd.to_numeric, errors='coerce')
    df_input.fillna(0, inplace=True)

    if hasattr(model, "feature_names_in_"):
        df_input = df_input[model.feature_names_in_]

    return df_input

# ------------------ SINGLE PREDICT ------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        txn = request.get_json()
        txn_id = txn.get('transaction_id_anonymous', 'UNKNOWN')

        # Step 1: Apply Rule Engine
        is_fraud, reason = rule_engine.apply_rules(txn)
        fraud_score = 0.0
        fraud_source = "rule" if is_fraud else "none"

        if is_fraud:
            # Use ML model to get fraud score (not decision)
            df_input = preprocess(txn)
            fraud_score = round(model.predict_proba(df_input)[0][1], 4)
        else:
            is_fraud = False
            reason = None

        result = {
            "transaction_id": txn_id,
            "is_fraud": is_fraud,
            "fraud_source": fraud_source,
            "fraud_reason": reason,
            "fraud_score": fraud_score
        }

        # Store input transaction in DB as JSON
        session = Session()
        session.add(TransactionDetails(
            transaction_id=txn_id,
            transaction_data=txn  # Store the full transaction data as JSON
        ))
        session.commit()

        # Store in DB for fraud transactions
        if is_fraud:
            session.add(FraudPrediction(**result))
            session.commit()

        session.close()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------ BATCH PREDICT ------------------
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        batch_data = request.get_json()
        if not isinstance(batch_data, list):
            return jsonify({'error': 'Expected list of transactions'}), 400

        results = []
        session = Session()

        for txn in batch_data:
            txn_id = txn.get('transaction_id_anonymous', 'UNKNOWN')

            # Preprocess the transaction
            df_input = preprocess(txn)

            # Predict fraud probability from model
            proba = model.predict_proba(df_input)[0][1]
            fraud_score = round(proba, 4)

            # Apply rule engine
            is_fraud, reason = rule_engine.apply_rules(txn)
            fraud_source = "rule" if is_fraud else "model"

            if not is_fraud:
                is_fraud = proba > 0.5
                fraud_source = "model"
                reason = "model_prediction" if is_fraud else None

            # Save result
            result = {
                "transaction_id": txn_id,
                "is_fraud": bool(is_fraud),
                "fraud_source": fraud_source,
                "fraud_reason": reason,
                "fraud_score": fraud_score
            }

            # Store input transaction in DB as JSON
            session.add(TransactionDetails(
                transaction_id=txn_id,
                transaction_data=txn  # Store the full transaction data as JSON
            ))

            # Save fraud prediction result in DB
            session.add(FraudPrediction(**result))

            results.append(result)

        session.commit()
        session.close()

        return jsonify(results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ------------------ REPORT FRAUD ------------------
@app.route('/report_fraud', methods=['POST'])
def report_fraud():
    try:
        txn = request.get_json()
        txn_id = txn.get('transaction_id_anonymous', 'UNKNOWN')

        session = Session()
        session.add(FraudReports(
            transaction_id=txn_id,
            reported_by_user=True,
            manual_fraud_label=1
        ))
        session.commit()
        session.close()

        return jsonify({
            "message": f"Transaction {txn_id} reported as fraud successfully!",
            "status": "logged"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------ VIEW ROUTES ------------------
@app.route('/all_predictions', methods=['GET'])
def get_all_predictions():
    session = Session()
    records = session.query(FraudPrediction).all()
    session.close()

    return jsonify([{
        "transaction_id": r.transaction_id,
        "is_fraud": r.is_fraud,
        "fraud_source": r.fraud_source,
        "fraud_reason": r.fraud_reason,
        "fraud_score": r.fraud_score
    } for r in records])

@app.route('/reported_frauds', methods=['GET'])
def get_reported_frauds():
    session = Session()
    records = session.query(FraudReports).all()
    session.close()

    return jsonify([{
        "transaction_id": r.transaction_id,
        "reported_by_user": r.reported_by_user,
        "manual_fraud_label": r.manual_fraud_label
    } for r in records])

# ------------------ RUN ------------------
if __name__ == '__main__':
    app.run(debug=True)
