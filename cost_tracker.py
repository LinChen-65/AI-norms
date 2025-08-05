
class Qwen30BTracker:
    def __init__(self, input_price_per_million=0.2, output_price_per_million=0.8):
        self.input_price_per_million = input_price_per_million
        self.output_price_per_million = output_price_per_million
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    def add_usage(self, input_tokens, output_tokens):
        """记录一次调用的 token 用量并计算费用"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        cost_input = (input_tokens / 1_000_000) * self.input_price_per_million
        cost_output = (output_tokens / 1_000_000) * self.output_price_per_million
        total_cost = cost_input + cost_output

        self.total_cost_usd += total_cost
        return total_cost

    def summary(self):
        """返回累计用量和总费用"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6)
        }
