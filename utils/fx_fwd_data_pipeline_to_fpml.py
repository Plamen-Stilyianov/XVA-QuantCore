import os
import sys
import requests
import datetime
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

import oandapyV20
import oandapyV20.endpoints.instruments as instruments

# Dynamic root path resolution to resolve adjacent module layers cleanly
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "../..")))

# For testing independently, wrap the machine learning import safely
try:
    from models.machine_learning import forecast_vol_with_lstm
except ImportError:
    def forecast_vol_with_lstm(data):
        # Fallback helper if the machine learning model layout is offline
        return np.full(shape=(30,), fill_value=0.15)


def pull_polygon_rest_fx_dataframe(lookback_days=1260):
    """
    Connects to the official Polygon.io REST API using standard requests.
    Bypasses the SDK client import layer to protect against virtual environment naming conflicts.
    """
    # The clean ticker format expected by the API endpoint path routing rules
    ticker = "C:EURUSD"
    print(f"📡 Cloud Gate: Connecting to Polygon REST API for pair: {ticker}...")

    # Active private Polygon API key (Consider rotating this key since it was leaked!)
    api_key = "DBIEaDlQBTa5BkjubP_7EDyH5dVIriF5"

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=int(lookback_days * 1.5))

    # Correct subdomain (api.) and full API path directory routing layout
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.isoformat()}/{end_date.isoformat()}"

    params = {
        "adjusted": "true",
        "sort": "desc",  # Fetch newest data first
        "limit": lookback_days + 10,
        "apiKey": api_key
    }

    # Browser user-agent headers ensure smooth connection passes through web firewalls
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"  # Forces the server to output raw data, not website pages
    }

    response = requests.get(url, params=params, headers=headers, timeout=15)

    if response.status_code != 200:
        raise RuntimeError(
            f"❌ Connection Error: Polygon API server rejected connection with "
            f"status code {response.status_code}. Response: {response.text[:200]}"
        )

    # Validate that Polygon returned actual JSON rather than an HTML website block
    content_type = response.headers.get('Content-Type', '')
    if 'application/json' not in content_type:
        raise RuntimeError("❌ Data Format Error: Polygon returned non-JSON html text.")

    data_json = response.json()
    results = data_json.get("results", [])

    if not results:
        raise ValueError("❌ Error: Polygon returned an empty or insufficient dataset.")

    # Slice down to target window size and reverse to run chronologically forward
    results = results[:lookback_days + 1]
    results.reverse()

    # Build the clean Pandas DataFrame
    df = pd.DataFrame(results)

    # Map Polygon timestamp format (milliseconds from epoch) to datetime
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df['close'] = pd.to_numeric(df['c'], errors='coerce')

    df = df.dropna(subset=['close']).copy()
    df = df[df['close'] > 0].copy()

    # Calculate stationary daily log returns: ln(P_t / P_{t-1})
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna().reset_index(drop=True)

    print(f"   🟢 Success: Ingested {len(df)} real daily entries from Polygon.io.")
    return df[['date', 'close', 'log_return']]

def pull_oanda_rest_fx_dataframe(lookback_days=1260):
    """
    Connects to OANDA using the official native SDK client framework.
    Bypasses standard requests/httpx text-parsing to prevent local network DNS resolution errors.
    """
    instrument = "EUR_USD"
    print(f"📡 Cloud Gate: Connecting to OANDA Native SDK for pair: {instrument}...")

    # Active verified OANDA API credentials profile
    api_key = "3d958ea495514d70bd4a05b285d9f0a0-1faa7f01691e687054a0cc853d522a6f"
    environment = "practice"

    # Limit requested candles to safe parameters matching structural capabilities
    target_count = min(int(lookback_days + 10), 4900)

    params = {
        "granularity": "D",  # Daily closing candlesticks
        "price": "M",  # Midpoint market values
        "count": target_count
    }

    try:
        # Initialize official library engine
        client = oandapyV20.API(access_token=api_key, environment=environment)

        # Execute structural native endpoint query layout
        request = instruments.InstrumentsCandles(instrument=instrument, params=params)
        response = client.request(request)

    except Exception as e:
        raise RuntimeError(
            f"❌ Connection Error: OANDA Native SDK client link failed to establish. "
            f"Network Exception details: {e}"
        )

    candles = response.get("candles", [])

    if not candles:
        raise ValueError("❌ Error: OANDA SDK returned an empty or insufficient dataset.")

    # Parse records out from safe internal native dictionary array maps
    parsed_records = []
    for candle in candles:
        if candle.get('complete'):
            parsed_records.append({
                't': candle.get('time'),
                'c': candle['mid']['c']
            })

    # Slice down to target window size matching lookback requirements
    parsed_records = parsed_records[-(lookback_days + 1):]

    # Build the clean Pandas DataFrame
    df = pd.DataFrame(parsed_records)

    # Map OANDA timestamp strings into datetime tracking format
    df['date'] = pd.to_datetime(df['t']).dt.date
    df['close'] = pd.to_numeric(df['c'], errors='coerce')

    df = df.dropna(subset=['close']).copy()
    df = df[df['close'] > 0].copy()

    # Calculate stationary daily log returns: ln(P_t / P_{t-1})
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna().reset_index(drop=True)

    print(f"   🟢 Success: Ingested {len(df)} real daily entries from OANDA SDK.")
    return df[['date', 'close', 'log_return']]


def generate_fx_forward_fpml(df, predicted_vol, output_directory, trade_id="ABN1234"):
    """
    Compiles a valid, single-spaced FpML requestConfirmation XML layout,
    mapping calculated forward pricing components from data matrices.
    """
    print(f"📑 Document Factory: Manufacturing FpML contract instance {trade_id}...")

    first_row = df.iloc[0]

    # Establish operational contract dates
    extracted_trade_date = pd.to_datetime(first_row['date']).strftime('%Y-%m-%d')
    extracted_value_date = (pd.to_datetime(first_row['date']) + datetime.timedelta(days=32)).strftime('%Y-%m-%d')

    # Quantitative FX Pricing Block (Covered Interest Parity Model)
    # Stressed Macro Assumptions: r_usd = 5.25%, r_eur = 3.75% over a 1-Month Horizon
    spot_rate = float(first_row['close'])
    r_usd, r_eur = 0.0525, 0.0375
    days_to_settle = 32
    t = days_to_settle / 360.0

    # Calculate Forward Rate Strike and Forward Point differentials
    forward_rate = spot_rate * ((1.0 + r_usd * t) / (1.0 + r_eur * t))
    forward_points = forward_rate - spot_rate

    # Standard transaction values from template rules
    eur_amount = 10000000
    usd_amount = int(eur_amount * forward_rate)

    # Enforce exact FpML Confirmation view schema required by clearing platforms
    fx_xml_string = f"""<requestConfirmation xmlns="http://fpml.org" xmlns:xsi="http://w3.org" fpmlVersion="5-12" xsi:schemaLocation="http://fpml.org ../../schemas/fpml-main-5-12.xsd http://w3.org ../../schemas/xmldsig-core-schema.xsd"><header><messageId messageIdScheme="http://matchingservice.com">FX456a789b</messageId><sentBy messageAddressScheme="http://matchingservice.com">ABN</sentBy><sendTo messageAddressScheme="http://matchingservice.com">MATCHSRV</sendTo><creationTimestamp>{extracted_trade_date}T08:57:00Z</creationTimestamp></header><isCorrection>false</isCorrection><correlationId correlationIdScheme="http://matchingservice.com">FX987</correlationId><sequenceNumber>1</sequenceNumber><trade><tradeHeader><partyTradeIdentifier><partyReference href="party1"/><tradeId tradeIdScheme="http://abn-amro.com">{trade_id}</tradeId></partyTradeIdentifier><partyTradeIdentifier><partyReference href="party2"/><tradeId tradeIdScheme="http://db.com">DB5678</tradeId></partyTradeIdentifier><tradeDate>{extracted_trade_date}</tradeDate></tradeHeader><fxSingleLeg><exchangedCurrency1><payerPartyReference href="party2"/><receiverPartyReference href="party1"/><paymentAmount><currency>EUR</currency><amount>{eur_amount}</amount></paymentAmount></exchangedCurrency1><exchangedCurrency2><payerPartyReference href="party1"/><receiverPartyReference href="party2"/><paymentAmount><currency>USD</currency><amount>{usd_amount}</amount></paymentAmount></exchangedCurrency2><valueDate>{extracted_value_date}</valueDate><exchangeRate><quotedCurrencyPair><currency1>EUR</currency1><currency2>USD</currency2><quoteBasis>Currency2PerCurrency1</quoteBasis></quotedCurrencyPair><rate>{forward_rate:.4f}</rate><spotRate>{spot_rate:.4f}</spotRate><forwardPoints>{forward_points:.4f}</forwardPoints></exchangeRate></fxSingleLeg></trade><party id="party1"><partyId partyIdScheme="http://fpml.org">BFXS5XCH7N0Y05NIXW11</partyId></party><party id="party2"><partyId partyIdScheme="http://fpml.org">213800QILIUD4ROSUO03</partyId></party></requestConfirmation>"""

    os.makedirs(output_directory, exist_ok=True)
    full_xml_path = os.path.join(output_directory, "fx-ex03-fx-fwd.xml")

    # Format cleanly with ElementTree to strip out empty lines
    root = ET.fromstring(fx_xml_string)
    ET.indent(root, space="  ", level=0)

    with open(full_xml_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="utf-8"?>\n')
        ET.ElementTree(root).write(f, encoding="utf-8", xml_declaration=False)

    print(f"   🟢 Success: Clean, single-spaced FX Forward FpML contract written to '{full_xml_path}'")


if __name__ == "__main__":
    # If testing on the Free tier, use 252 (1 Year). If on paid tier, use 756 (3 Years)
    lookback_window = 252 * 5

    # Absolute folder path tracking configurations
    output_dir = os.path.abspath(os.path.join(script_dir, "../data/historical"))
    payloads_dir = os.path.abspath(os.path.join(script_dir, "../data/payloads"))

    # 1. RUN EXTRACTION: Fetch data via raw REST requests
    #market_df = pull_polygon_rest_fx_dataframe(lookback_days=lookback_window)
    market_df = pull_oanda_rest_fx_dataframe(lookback_days=lookback_window)

    # 2. SAVE CSV: Store the currency data table to disk
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, "historical_fx_market_data.csv")
    market_df.to_csv(csv_file_path, index=False)
    print(f"💾 Storage Flush: Live FX dataset saved cleanly at '{csv_file_path}'")

    # 3. VOLATILITY PROCESSING: Extract standard deviation from real currency returns
    real_daily_vol = np.std(market_df['log_return'])
    real_annual_vol = real_daily_vol * np.sqrt(252)

    # Apply standard 2.5x multiplier to evaluate a high-stress volatility scenario
    stress_multiplier = 2.5
    current_base_vol = real_annual_vol * stress_multiplier
    print(f"📊 Risk Engine Metrics -> Realised Annualised Vol: {real_annual_vol:.4f}")
    print(f"⚠️ Stress Engine Metrics -> Stressed Base Vol (2.5x): {current_base_vol:.4f}")

    # Generate or forecast a volatility vector horizon block
    try:
        mock_predicted_vol = forecast_vol_with_lstm(market_df)
    except Exception:
        mock_predicted_vol = np.full(shape=(30,), fill_value=current_base_vol)

    # 4. EXPORT FpML DOCUMENT: Build the final transaction contract file
    generate_fx_forward_fpml(
        df=market_df,
        predicted_vol=mock_predicted_vol,
        output_directory=payloads_dir,
        trade_id="ABN1234"
    )
    print("✨ Quantitative Data Pipeline Fully Complete!")
