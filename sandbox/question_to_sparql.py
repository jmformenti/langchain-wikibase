import os
import argparse
import spacy
from wikibaseintegrator import wbi_helpers
from wikibaseintegrator.wbi_config import config as wbi_config
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


WB_LANGUAGE = 'en'
WB_LIMIT = 10
WB_USER_AGENT = 'MyWikibaseBot/1.0'

wbi_config['USER_AGENT'] = 'MyWikibaseBot/1.0'

def getQItem(name: str) -> str:
  """Returns the Q item from my wikibase."""

  data = {
    'action': 'wbsearchentities',
    'search': name,
    'type': 'item',
    'language': WB_LANGUAGE,
    'limit': WB_LIMIT
  }
  result = wbi_helpers.mediawiki_api_call_helper(data=data, allow_anonymous=True)
  if result['search']:
    return result['search'][0]['id']
  else:
    return 'Item not found by this name, try another name.'

def getProperty(name: str) -> str:
  """Returns the property from my wikibase."""

  data = {
    'action': 'wbsearchentities',
    'search': name,
    'type': 'property',
    'language': WB_LANGUAGE,
    'limit': WB_LIMIT
  }
  result = wbi_helpers.mediawiki_api_call_helper(data=data, allow_anonymous=True)
  if result['search']:
    return result['search'][0]['id']
  else:
    return 'Property not found by this name, try another name.'

def insertStr(original, substr, position):
  return f"{original[:position]} ({substr}){original[position:]}", len(substr) + 3

def normalize(original):
  return original.replace("'s", "")

def generate_sparql(text):
  prompt = PromptTemplate.from_template("""
  Given the next question and the info between brackets, write the SPARQL query for a wikibase:
  {question}
  Return only the SPARQL query without any explanation.
  """);
  openai_api_base = None
  if 'OPENAI_API_BASE' in os.environ:
    openai_api_base = os.environ['OPENAI_API_BASE']

  llm = ChatOpenAI(openai_api_base=openai_api_base, openai_api_key="ignored", model="mixtral", temperature=0.1)
  return llm.invoke(prompt.format(question=text))

def enrich_text(text, debug):
  nlp = spacy.load("en_core_web_sm")

  doc = nlp(text)

  print(f'# Original question\n{text}')

  elements = {}

  # Identify properties
  if debug:
    print(f'\n## Tokens')
  for token in doc:
    if debug:
      print(f"Token '{token}' type {token.pos_} with lemma '{token.lemma_}' at {token.idx}")
    if token.pos_ == "NOUN":
      elements[token.idx] = { 'element': token, 'type': 'PROPERTY' }

  # Identify entities
  if debug:
    print(f'\n## Entities')
  for ent in doc.ents:
    if debug:
      print(f"Entity '{ent}' at ({ent.start_char - ent.sent.start_char}, {ent.end_char-ent.sent.start_char})")
    start_pos = ent.start_char - ent.sent.start_char
    elements[start_pos] = { 'element': ent, 'type': 'ENTITY' }

  # Enrich original text
  text_enriched = text
  acc_num_new_chars = 0

  sorted_elements_by_pos = dict(sorted(elements.items()))

  for key, item in sorted_elements_by_pos.items():
    element_type = item['type']
    element = item['element']

    semantic_info = ''
    if element_type == 'PROPERTY':
      if element.lemma_ == "name":
        semantic_info = 'label'
      else:
        semantic_info = getProperty(element.text)
    elif element_type == 'ENTITY':
      semantic_info = getQItem(normalize(element.text))
    else:
      raise ValueError(f"'{item.type}' not supported.")
    
    pos = key + acc_num_new_chars + len(element.text)

    text_enriched, num_new_chars = insertStr(text_enriched, semantic_info, pos)
    acc_num_new_chars += num_new_chars

  return text_enriched


parser = argparse.ArgumentParser(description='SPARQL query generator.')
parser.add_argument('--question', type=str, required=True, help='Your question.')
parser.add_argument('--debug', action='store_true', help='Show debug info.')

args = parser.parse_args()

# Enrich original text
text_enriched = enrich_text(args.question, args.debug)
print(f'\n# Enriched question\n{text_enriched}')

# Generate SPARQL query using LLM
sparql_query = generate_sparql(text_enriched)
print(f'\n# SPARQL query\n{sparql_query.content.strip()}')
