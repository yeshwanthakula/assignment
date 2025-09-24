import re
try:
    from guardrails.hub import DetectPII
    import guardrails as gd
except ImportError:
    print("Guardrails not installed...")
    pass
import spacy


def gaurd_rail_output(text):
    guard = gd.Guard().use(DetectPII(pii_entities="pii", on_fail="fix"))
    output = guard.parse(
        llm_output=text,
    )
    return output.raw_llm_output , output.validated_output



def mask_phone_numbers(original, masked):
    placeholder = "<PHONE_NUMBER>"
    numbers = re.findall(r'\d{7,}', original)
    masked_result = masked
    offset = 0

    for number in numbers:
        masked_number = f"xxxx{number[-3:]}"
        idx = masked_result.find(placeholder, offset)
        if idx == -1:
            break
        masked_result = (
            masked_result[:idx] +
            masked_number +
            masked_result[idx + len(placeholder):]
        )
        offset = idx + len(masked_number)
    return masked_result



def predict_escalation(nlp, incident_text):
    """Predict escalation level for given incident text"""
    doc = nlp(incident_text)
    
    # Get prediction scores
    scores = doc.cats
    
    # Find the highest scoring category
    predicted_level = max(scores, key=scores.get)
    confidence = scores[predicted_level]
    
    if confidence>0.98:
        print("confidence",confidence)
        return  predicted_level
    else :
       return None
    
    # return {
    #     "level": predicted_level,
    #     "confidence": round(confidence, 3),
    #     "all_scores": {k: round(v, 3) for k, v in scores.items()}
    # }


def load_model(model_path="escalation_model"):
    """Load a saved model"""
    loaded_nlp = spacy.load(model_path)
    return loaded_nlp


def escalate(level):
    print(f"Escalation level: {level}")
    print("-" * 80)
    if level == "P0":
          print("Trigeering page on-call within 5 minutes .")
    elif level == "P1":
        print( "escalating to Incident Manager within 30 minutes.")
    elif level == "P2":
        return print("open Jira and respond within 4 business hours.")