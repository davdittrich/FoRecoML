import os
import re
import time

file_path = '/home/dd/.agents/skills/infographics/scripts/generate_infographic_ai.py'
with open(file_path + '.bak', 'r') as f:
    content = f.read()

# Replace __init__ and _make_request logic
old_block_pattern = r'    def __init__\(self, api_key: Optional\[str\] = None, verbose: bool = False\):.*?    def _extract_image_from_response'
new_block = """    def __init__(self, api_key: Optional[str] = None, verbose: bool = False):
        \"\"\"Initialize the generator with multi-provider support.\"\"\"
        _load_env_file()
        
        self.openrouter_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Determine default provider
        if self.google_api_key:
            self.provider = "google"
            self.api_key = self.google_api_key
        elif self.anthropic_api_key:
            self.provider = "anthropic"
            self.api_key = self.anthropic_api_key
        elif self.openrouter_api_key:
            self.provider = "openrouter"
            self.api_key = self.openrouter_api_key
        else:
            self.provider = "openrouter"
            self.api_key = None
            
        self.verbose = verbose
        self._last_error = None
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Model defaults
        if self.provider == "google":
            self.image_model = "gemini-1.5-pro"
            self.review_model = "gemini-1.5-pro"
        elif self.provider == "anthropic":
            self.image_model = "claude-3-5-sonnet-20241022"
            self.review_model = "claude-3-5-sonnet-20241022"
        else:
            self.image_model = "google/gemini-3-pro-image-preview"
            self.review_model = "google/gemini-3.1-pro-preview"
        
    def _log(self, message: str):
        \"\"\"Log message if verbose mode is enabled.\"\"\"
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    # ========== RESEARCH METHODS ==========
    
    def research_topic(self, topic: str, infographic_type: Optional[str] = None) -> Dict[str, Any]:
        \"\"\"Research a topic using Perplexity Sonar Pro (via OpenRouter) or Gemini.\"\"\"
        self._log(f"Researching topic: {topic}")
        import requests
        type_context = ""
        if infographic_type == "statistical": type_context = "Focus on statistics and data."
        elif infographic_type == "timeline": type_context = "Focus on key dates."
        
        messages = [
            {\"role\": \"system\", \"content\": \"You are an expert research assistant.\"},
            {\"role\": \"user\", \"content\": f\"Provide key facts for an infographic about: {topic}. {type_context}\"}
        ]
        
        try:
            if self.provider == \"openrouter\" and self.openrouter_api_key:
                headers = {\"Authorization\": f\"Bearer {self.openrouter_api_key}\", \"Content-Type\": \"application/json\"}
                payload = {\"model\": \"perplexity/sonar-pro\", \"messages\": messages, \"max_tokens\": 2000}
                response = requests.post(f\"{self.base_url}/chat/completions\", headers=headers, json=payload, timeout=60)
                if response.status_code == 200:
                    result = response.json()
                    return {\"success\": True, \"content\": result[\"choices\"][0][\"message\"][\"content\"], \"sources\": result.get(\"search_results\", [])}
            
            response = self._make_request(self.review_model, messages)
            return {\"success\": True, \"content\": response[\"choices\"][0][\"message\"][\"content\"], \"sources\": [], \"model\": self.review_model}
        except Exception as e:
            return {\"success\": False, \"error\": str(e)}

    def web_search(self, query: str) -> Dict[str, Any]:
        \"\"\"Perform a quick web search.\"\"\"
        self._log(f"Web search: {query}")
        messages = [{\"role\": \"user\", \"content\": f\"Search for current information about: {query}\"}]
        try:
            response = self._make_request(self.review_model, messages)
            return {\"success\": True, \"content\": response[\"choices\"][0][\"message\"][\"content\"]}
        except Exception as e:
            return {\"success\": False, \"error\": str(e)}

    def _make_request(self, model: str, messages: List[Dict[str, Any]], 
                     modalities: Optional[List[str]] = None) -> Dict[str, Any]:
        \"\"\"Dispatch request to the appropriate provider.\"\"\"
        provider = self.provider
        if \"/\" in model and not model.startswith(\"gemini/\") and not model.startswith(\"claude/\"):
            provider = \"openrouter\"
        elif \"gemini\" in model.lower():
            provider = \"google\" if self.google_api_key else \"openrouter\"
        elif \"claude\" in model.lower():
            provider = \"anthropic\" if self.anthropic_api_key else \"openrouter\"
            
        import requests
        if provider == \"google\": return self._make_google_request(model, messages, modalities)
        elif provider == \"anthropic\": return self._make_anthropic_request(model, messages)
        else: return self._make_openrouter_request(model, messages, modalities)

    def _make_openrouter_request(self, model: str, messages: List[Dict[str, Any]], 
                                modalities: Optional[List[str]] = None) -> Dict[str, Any]:
        \"\"\"Make a request to OpenRouter API.\"\"\"
        import requests
        key = self.openrouter_api_key or self.api_key
        if not key: raise ValueError(\"OPENROUTER_API_KEY required\")
        headers = {\"Authorization\": f\"Bearer {key}\", \"Content-Type\": \"application/json\"}
        payload = {\"model\": model, \"messages\": messages}
        if modalities: payload[\"modalities\"] = modalities
        response = requests.post(f\"{self.base_url}/chat/completions\", headers=headers, json=payload, timeout=120)
        if response.status_code != 200: raise RuntimeError(f\"OpenRouter failed: {response.text}\")
        return response.json()

    def _make_google_request(self, model: str, messages: List[Dict[str, Any]], 
                            modalities: Optional[List[str]] = None) -> Dict[str, Any]:
        \"\"\"Make a request directly to Google's Gemini API.\"\"\"
        import requests
        if not self.google_api_key: raise ValueError(\"GOOGLE_API_KEY required\")
        if \"/\" in model: model = model.split(\"/\")[-1]
        url = f\"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.google_api_key}\"
        contents = []
        for msg in messages:
            role = \"user\" if msg[\"role\"] == \"user\" else \"model\"
            parts = []
            content = msg[\"content\"]
            if isinstance(content, str): parts.append({\"text\": content})
            elif isinstance(content, list):
                for part in content:
                    if part[\"type\"] == \"text\": parts.append({\"text\": part[\"text\"]})
                    elif part[\"type\"] == \"image_url\":
                        img_url = part[\"image_url\"][\"url\"]
                        if \",\" in img_url:
                            mime = img_url.split(\":\")[1].split(\";\")[0]
                            data = img_url.split(\",\")[1]
                            parts.append({\"inline_data\": {\"mime_type\": mime, \"data\": data}})
            contents.append({\"role\": role, \"parts\": parts})
        response = requests.post(url, json={\"contents\": contents}, timeout=120)
        if response.status_code != 200: raise RuntimeError(f\"Google API failed: {response.text}\")
        result = response.json()
        try:
            cand = result[\"candidates\"][0]
            text = cand[\"content\"][\"parts\"][0][\"text\"]
            images = []
            for part in cand[\"content\"][\"parts\"]:
                if \"inline_data\" in part:
                    images.append({\"type\": \"image_url\", \"image_url\": {\"url\": f\"data:{part['inline_data']['mime_type']};base64,{part['inline_data']['data']}\"}})
            return {\"choices\": [{\"message\": {\"role\": \"assistant\", \"content\": text, \"images\": images}}]}
        except: raise RuntimeError(f\"Unexpected Google response: {result}\")

    def _make_anthropic_request(self, model: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        \"\"\"Make a request directly to Anthropic's Claude API.\"\"\"
        import requests
        if not self.anthropic_api_key: raise ValueError(\"ANTHROPIC_API_KEY required\")
        if \"/\" in model: model = model.split(\"/\")[-1]
        headers = {\"x-api-key\": self.anthropic_api_key, \"anthropic-version\": \"2023-06-01\", \"content-type\": \"application/json\"}
        anth_msgs = []
        system = \"\"
        for msg in messages:
            if msg[\"role\"] == \"system\": system = msg[\"content\"]
            else:
                role = \"user\" if msg[\"role\"] == \"user\" else \"assistant\"
                content = msg[\"content\"]
                if isinstance(content, list):
                    new_c = []
                    for p in content:
                        if p[\"type\"] == \"text\": new_c.append({\"type\": \"text\", \"text\": p[\"text\"]})
                        elif p[\"type\"] == \"image_url\":
                            url = p[\"image_url\"][\"url\"]
                            if \",\" in url:
                                mime = url.split(\":\")[1].split(\";\")[0]
                                data = url.split(\",\")[1]
                                new_c.append({\"type\": \"image\", \"source\": {\"type\": \"base64\", \"media_type\": mime, \"data\": data}})
                    content = new_c
                anth_msgs.append({\"role\": role, \"content\": content})
        payload = {\"model\": model, \"max_tokens\": 4096, \"messages\": anth_msgs}
        if system: payload[\"system\"] = system
        response = requests.post(\"https://api.anthropic.com/v1/messages\", headers=headers, json=payload, timeout=120)
        if response.status_code != 200: raise RuntimeError(f\"Anthropic failed: {response.text}\")
        result = response.json()
        text = \"\".join([p[\"text\"] for p in result.get(\"content\", []) if p[\"type\"] == \"text\"])
        return {\"choices\": [{\"message\": {\"role\": \"assistant\", \"content\": text}}]}

    def _extract_image_from_response"""

content = re.sub(old_block_pattern, new_block, content, flags=re.DOTALL)

# Update main
main_old = r'    # Check for API key.*?sys\.exit\(1\)'
main_new = """    # Check for API key
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(\"Error: No API key found (OPENROUTER_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY)\")
        print(\"\\nSet one with:\")
        print(\"  export GOOGLE_API_KEY='your_api_key'\")
        print(\"  export ANTHROPIC_API_KEY='your_api_key'\")
        print(\"  export OPENROUTER_API_KEY='your_api_key'\")
        sys.exit(1)"""

content = re.sub(main_old, main_new, content, flags=re.DOTALL)

with open(file_path, 'w') as f:
    f.write(content)
