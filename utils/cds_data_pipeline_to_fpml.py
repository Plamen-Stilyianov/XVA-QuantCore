import os
import requests
import pandas as pd
import numpy as np
import datetime
import xml.etree.ElementTree as ET


def pull_fred_credit_dataframe(lookback_days=1260):
    """
    Connects to the official FRED API using your key to fetch real historical
    daily records for the ICE BofA BBB US Corporate Index Option-Adjusted Spread.
    This provides the precise credit risk volatility needed for CDS modeling.
    """
    # Canonical corporate credit spread index on FRED
    series_id = "BAMLC0A4CBBB"
    print(f"📡 Cloud Gate: Connecting to FRED API for credit series: {series_id}...")

    # Your provided active FRED API key
    api_key = "50c4a3f98ed3f3a778cd60d453afbc97"

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",  # Fetch the newest entries first
        "limit": lookback_days + 1
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(url, params=params, headers=headers, timeout=15)

    if response.status_code != 200:
        raise RuntimeError(
            f"❌ Connection Error: FRED API server rejected connection with "
            f"status code {response.status_code}."
        )

    content_type = response.headers.get('Content-Type', '')
    if 'application/json' not in content_type:
        raise RuntimeError("❌ Data Format Error: FRED returned non-JSON data.")

    data_json = response.json()
    observations = data_json.get("observations", [])

    if len(observations) <= 1:
        raise ValueError("❌ Error: FRED returned an empty dataset.")

    observations.reverse()

    df = pd.DataFrame(observations)
    df['date'] = pd.to_datetime(df['date'])

    # Process observations and clear empty holiday placeholders ('.')
    df['close'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['close']).copy()
    df = df[df['close'] > 0].copy()

    # Calculate stationary daily credit log returns: ln(P_t / P_{t-1})
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna().reset_index(drop=True)

    print(f"   🟢 Success: Ingested {len(df)} real daily entries from FRED.")
    return df[['date', 'close', 'log_return']]


def forecast_vol_with_lstm(num_steps: int, base_vol: float) -> np.ndarray:
    """ LSTM Volatility Engine Pass (Karasan Chapter 4) """
    vol_curve = np.zeros(num_steps)
    h_t, c_t = 0.05, 0.01
    w_f, b_f = 0.85, -0.1
    w_i, b_i = 0.45, 0.2
    w_o, b_o = 0.65, 0.1
    w_c, b_c = 0.50, 0.0

    for t in range(num_steps):
        f_g = 1.0 / (1.0 + np.exp(-(w_f * h_t + b_f)))
        i_g = 1.0 / (1.0 + np.exp(-(w_i * h_t + b_i)))
        c_tilde = np.tanh(w_c * h_t + b_c)
        c_t = f_g * c_t + i_g * c_tilde
        o_g = 1.0 / (1.0 + np.exp(-(w_o * h_t + b_o)))
        h_t = o_g * np.tanh(c_t)
        vol_curve[t] = base_vol + (h_t * 0.005)
    return vol_curve


def generate_cds_fpml(df, predicted_vol, output_directory, trade_id="37264"):
    """
    Compiles a valid, single-spaced FpML dataDocument XML layout for a Credit Default Swap,
    synchronising contractual lifecycle headers with real timeline matrices.
    """
    print(f"📑 Document Factory: Manufacturing FpML contract instance {trade_id}...")

    first_row = df.iloc[0]
    last_row = df.iloc[-1]

    extracted_trade_date = pd.to_datetime(first_row['date']).strftime('%Y-%m-%d')
    extracted_effective_date = (pd.to_datetime(first_row['date']) + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    extracted_termination_date = (pd.to_datetime(last_row['date']) + datetime.timedelta(days=5 * 365)).strftime(
        '%Y-%m-%d')

    # Map the parsed initial credit spread value directly into fixed coupon legs (e.g., 2.15% -> 0.0215)
    initial_credit_spread = float(first_row['close']) / 100.0
    peak_stressed_vol = float(np.max(predicted_vol))
    notional_amount = 5000000.00

    cds_xml_string = f"""<dataDocument xmlns="http://www.fpml.org/FpML-5/confirmation" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" fpmlVersion="5-12" xsi:schemaLocation="http://www.fpml.org/FpML-5/confirmation ../../schemas/fpml-main-5-12.xsd http://www.w3.org/2000/09/xmldsig# ../../schemas/xmldsig-core-schema.xsd"><trade><tradeHeader><partyTradeIdentifier><partyReference href="party1"/><tradeId tradeIdScheme="http://www.swapswire.com/spec/2001/trade-id-1-0">{trade_id}</tradeId></partyTradeIdentifier><partyTradeIdentifier><partyReference href="party2"/><tradeId tradeIdScheme="http://www.swapswire.com/spec/2001/trade-id-1-0">{trade_id}</tradeId></partyTradeIdentifier><tradeDate>{extracted_trade_date}</tradeDate></tradeHeader><creditDefaultSwap><generalTerms><effectiveDate><unadjustedDate>{extracted_effective_date}</unadjustedDate><dateAdjustments><businessDayConvention>NONE</businessDayConvention></dateAdjustments></effectiveDate><scheduledTerminationDate><unadjustedDate>{extracted_termination_date}</unadjustedDate><dateAdjustments><businessDayConvention>MODFOLLOWING</businessDayConvention><businessCenters><businessCenter>GBLO</businessCenter><businessCenter>USNY</businessCenter></businessCenters></dateAdjustments></scheduledTerminationDate><buyerPartyReference href="party2"/><sellerPartyReference href="party1"/><dateAdjustments><businessDayConvention>MODFOLLOWING</businessDayConvention><businessCenters><businessCenter>GBLO</businessCenter><businessCenter>USNY</businessCenter></businessCenters></dateAdjustments><referenceInformation><referenceEntity id="referenceEntity">Agrium Inc.008HA7</referenceEntity><referenceObligation><bond><instrumentId instrumentIdScheme="http://www.fpml.org/coding-scheme/external/instrument-id-CUSIP">008916AB4</instrumentId><couponRate>0.077</couponRate><maturity>2017-02-01</maturity></bond><primaryObligorReference href="referenceEntity"/></referenceObligation><referencePrice>1.0</referencePrice></referenceInformation></generalTerms><feeLeg><periodicPayment><paymentFrequency><periodMultiplier>3</periodMultiplier><period>M</period></paymentFrequency><firstPaymentDate>{extracted_effective_date}</firstPaymentDate><rollConvention>6</rollConvention><fixedAmountCalculation><calculationAmount><currency>USD</currency><amount>{notional_amount}</amount></calculationAmount><fixedRate>{initial_credit_spread:.4f}</fixedRate><dayCountFraction>ACT/360</dayCountFraction></fixedAmountCalculation></periodicPayment></feeLeg><protectionTerms><calculationAmount><currency>USD</currency><amount>{notional_amount}</amount></calculationAmount><creditEvents><bankruptcy>true</bankruptcy><failureToPay><applicable>true</applicable><paymentRequirement><currency>USD</currency><amount>1000000.0</amount></paymentRequirement></failureToPay><restructuring><applicable>true</applicable><restructuringType>ModR</restructuringType></restructuring><defaultRequirement><currency>USD</currency><amount>10000000.0</amount></defaultRequirement><creditEventNotice><notifyingParty><buyerPartyReference href="party2"/><sellerPartyReference href="party1"/></notifyingParty><publiclyAvailableInformation><standardPublicSources>true</standardPublicSources><specifiedNumber>2</specifiedNumber></publiclyAvailableInformation></creditEventNotice></creditEvents><obligations><category>BorrowedMoney</category></obligations></protectionTerms><physicalSettlementTerms><settlementCurrency>USD</settlementCurrency><physicalSettlementPeriod><maximumBusinessDays>30</maximumBusinessDays></physicalSettlementPeriod><deliverableObligations><accruedInterest>false</accruedInterest><category>BondOrLoan</category><notSubordinated>true</notSubordinated><specifiedCurrency><applicable>true</applicable></specifiedCurrency><notContingent>true</notContingent><assignableLoan><applicable>true</applicable></assignableLoan><consentRequiredLoan><applicable>true</applicable></consentRequiredLoan><transferable>true</transferable><maximumMaturity><periodMultiplier>30</periodMultiplier><period>Y</period></maximumMaturity><notBearer>true</notBearer></deliverableObligations><escrow>true</escrow></physicalSettlementTerms></creditDefaultSwap><calculationAgent><calculationAgentPartyReference href="party1"/></calculationAgent><calculationAgentBusinessCenter>USNY</calculationAgentBusinessCenter><documentation><masterAgreement><masterAgreementType>ISDA</masterAgreementType><masterAgreementVersion>1992</masterAgreementVersion></masterAgreement><contractualDefinitions>ISDA1999Credit</contractualDefinitions></documentation></trade><party id="party1"><partyId partyIdScheme="http://www.fpml.org/coding-scheme/external/iso17442">254900BIAQJIUV6DLE92</partyId><partyName>XYZ Bank</partyName></party><party id="party2"><partyId partyIdScheme="http://www.fpml.org/coding-scheme/external/iso17442">549300VBWWV6BYQOWM67</partyId><partyName>ABC Bank</partyName></party></dataDocument>"""

    os.makedirs(output_directory, exist_ok=True)
    full_xml_path = os.path.join(output_directory, "cd-ex10-long-us-corp-fixreg.xml")

    # Native ElementTree serialization ensures clean layout mapping without double newlines
    root = ET.fromstring(cds_xml_string)
    ET.indent(root, space="  ", level=0)

    with open(full_xml_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="utf-8"?>\n')
        ET.ElementTree(root).write(f, encoding="utf-8", xml_declaration=False)

    print(f"   🟢 Success: Clean, single-spaced CDS FpML contract written to '{full_xml_path}'")


if __name__ == "__main__":
    lookback_window = 252 * 5
    num_steps = 252 * 5

    output_dir = "../data/historical"
    payloads_dir = "../data/payloads"

    # 1. RUN EXTRACTION: Ingest historical Corporate Bond Spreads from FRED
    market_df = pull_fred_credit_dataframe(lookback_days=lookback_window)

    # 2. SAVE CSV: Store the data table using your synchronized naming pattern
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, "historical_cds_market_data.csv")
    market_df.to_csv(csv_file_path, index=False)
    print(f"💾 Storage Flush: Live CDS dataset saved cleanly at '{csv_file_path}'")

    # 3. VOLATILITY PROCESSING: Extract standard deviation from real credit returns
    real_daily_vol = np.std(market_df['log_return'])
    real_annual_vol = real_daily_vol * np.sqrt(252)

    # Apply your standard 2.5x multiplier to execute the high-stress test scenario
    stress_multiplier = 2.5
    current_base_vol = real_annual_vol * stress_multiplier
    print(f"📊 Real Annual Vol: {real_annual_vol * 100:.2f}% | Stressed Base Vol: {current_base_vol * 100:.2f}%")

    # 4. MODEL DESIGN: Route metrics through the LSTM forecasting layer
    predicted_vol_curve = forecast_vol_with_lstm(num_steps, base_vol=current_base_vol)

    # 5. COMPILE CONTRACT payLoad: Output perfectly formatted transaction structure
    generate_cds_fpml(df=market_df, predicted_vol=predicted_vol_curve, output_directory=payloads_dir)
    print("\n🏁 Core Process Complete: CDS data pipeline execution successful.")