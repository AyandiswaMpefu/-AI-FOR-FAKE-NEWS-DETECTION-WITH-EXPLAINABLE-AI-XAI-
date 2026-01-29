from django import forms
from urllib.parse import urlparse

class UrlForm(forms.Form):
    url = forms.URLField(label="News article URL", max_length=500)

    def clean_url(self):
        u = self.cleaned_data["url"]
        parsed = urlparse(u)
        if parsed.scheme not in ("http", "https"):
            raise forms.ValidationError("Only http/https URLs allowed")
        return u
