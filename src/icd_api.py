import re        # Regular expressions
import requests  # HTTP requests
from config import (
    ICD_CLIENT_ID, ICD_CLIENT_SECRET,
    ICD_TOKEN_URL, ICD_SEARCH_URL,
    ICD_SCORE_THRESHOLD, REQUEST_TIMEOUT
)

def get_icd_token() -> str:
    """Request an access token for ICD-11 API.

    Returns:
        str: Access token for ICD-11 API, or None if request fails.
    """
    # Prepare the data for token request
    data = {
        'client_id': ICD_CLIENT_ID,
        'client_secret': ICD_CLIENT_SECRET,
        'scope': 'icdapi_access',
        'grant_type': 'client_credentials'
    }

    # Make the POST request to get the token
    try:
        response = requests.post(ICD_TOKEN_URL, data=data)
        response.raise_for_status()
        return response.json().get('access_token')
    
    # Handle request errors
    except requests.RequestException as e:
        print(f"[ICD ERROR] Token request failed: {e}")
        return None

def icd_search(term: str, question_type: str) -> list[dict]:
    """Search ICD-11 for a term and return relevant entities.

    Args:
        term (str): The search term to query ICD-11.
        question_type (str): The type of question ('clinical', 'definition', etc.).

    Returns:
        list[dict]: A list of matching ICD-11 entities, or an empty list if no matches found.
    """
    # Get the access token and construct headers
    token = get_icd_token()
    headers = {
        'Authorization': f'Bearer {token}',
        'API-Version': 'v2',
        'Accept-Language': 'en'
    }

    # Perform the search request
    try:
        response = requests.get(f"{ICD_SEARCH_URL}?q={term}", headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        entities = response.json().get('destinationEntities', [])
        if not entities:
            return []

        sorted_entities = sorted(entities, key=lambda x: float(x.get('score', 0)), reverse=True)
        best_match = sorted_entities[0]
        if float(best_match.get('score', 0)) < ICD_SCORE_THRESHOLD:
            print("[ICD] Best match score below threshold.")
            return []

        # If the question type is 'clinical' or 'definition', fetch full entity details
        if question_type in ("clinical", "definition"):
            entity_url = best_match.get("id")
            if not entity_url:
                return []
            full_resp = requests.get(entity_url, headers=headers, timeout=REQUEST_TIMEOUT)
            full_resp.raise_for_status()
            full_entity = full_resp.json()

            # If definition is missing, try to get it from parent entity
            if not full_entity.get("definition"):
                parent_urls = full_entity.get("parent", [])
                if parent_urls:
                    parent_resp = requests.get(parent_urls[0], headers=headers, timeout=REQUEST_TIMEOUT)
                    parent_resp.raise_for_status()
                    parent_entity = parent_resp.json()
                    parent_def = parent_entity.get("definition", {})
                    if isinstance(parent_def, dict):
                        full_entity["definition"] = parent_def.get("@value", "") or parent_def.get("value", "")
                    elif isinstance(parent_def, str):
                        full_entity["definition"] = parent_def

            if not full_entity.get("definition"):
                full_entity["definition"] = "No definition available from ICD-11."
            return [full_entity]

        # Otherwise, return only title and code
        else:
            return [best_match]

    # Handle request errors
    except requests.RequestException as e:
        print(f"[ICD ERROR] Search failed: {e}")
        return []

def log_icd_result(label, result, seen_titles = None) -> None:
    """Log a readable version of the ICD result.
    
    Args:
        label (str): The label or term used for the search.
        result (dict): The ICD result dictionary.
    """
    # Handle both dict and str cases for title extraction
    raw_title = result.get("title", "Title unavailable")  
    title = raw_title.get("@value", raw_title.get("value", "Title unavailable")) if isinstance(raw_title, dict) else raw_title
    title = re.sub(r"<[^>]+>", "", title or "").strip()  # Remove HTML tags and whitespace
    if seen_titles is not None:
        if title.lower() in seen_titles:
            return  # Avoid duplicate log
        seen_titles.add(title.lower())
        
    # Handle code extraction
    code = result.get("code", result.get("theCode", "Code unavailable"))

    # Handle definition extraction
    raw_def = result.get("definition", {})
    definition = raw_def.get("value", raw_def.get("@value", "")) if isinstance(raw_def, dict) else raw_def or ""

    # Log the formatted output
    print(f"[ICD DEBUG] MAPPING: {label} -> {title} (Code: {code})\nDefinition: {definition}\n")