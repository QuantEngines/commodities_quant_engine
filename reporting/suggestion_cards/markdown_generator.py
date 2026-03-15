from ...data.models import Suggestion

class MarkdownGenerator:
    """Generate markdown reports from suggestions."""
    
    @staticmethod
    def generate_card(suggestion: Suggestion) -> str:
        """Generate markdown suggestion card."""
        return suggestion.to_markdown()
    
    @staticmethod
    def generate_table(suggestions: list[Suggestion]) -> str:
        """Generate markdown table of suggestions."""
        if not suggestions:
            return "No suggestions available."
        
        header = "| Commodity | Category | Direction | Confidence | Regime |\n|-----------|----------|-----------|------------|--------|"
        rows = []
        
        for s in suggestions:
            row = f"| {s.commodity} | {s.final_category} | {s.preferred_direction} | {s.confidence_score:.2f} | {s.regime_label} |"
            rows.append(row)
        
        return header + "\n" + "\n".join(rows)