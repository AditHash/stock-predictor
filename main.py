from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch
from phi.tools.yfinance import YFinanceTools
from phi.model.aws.claude import Claude  # For AWS Bedrock Claude model
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Ensure AWS credentials are set via environment variables or CLI configuration
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")  # Default to us-east-1

# Initialize FastAPI app
app = FastAPI()

# Create an agent with combined capabilities, using Claude from AWS Bedrock
agent = Agent(
    model=Claude(id="anthropic.claude-3-5-sonnet-20240620-v1:0"),  # Claude model from AWS Bedrock
    # model=Claude(id="anthropic.claude-3-haiku-20240307-v1:0"),
    tools=[
        DuckDuckGo(),
        GoogleSearch(),
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True),
    ],
    show_tool_calls=False,
    description=(
        "You are an investment advisor. You research stock prices, analyst recommendations, "
        "and stock fundamentals, and you find relevant information from DuckDuckGo and Google."
    ),
    instructions=[
        "When a user queries about stocks, search for relevant information using DuckDuckGo and Google.",
        "Use YFinance to gather stock prices, analyst recommendations, and stock fundamentals.",
        "Combine all findings and provide a detailed markdown-formatted response with tables and clear conclusions.",
        "Dont Start with Thank you or thanks for providing data, Just start with here is what i found or here is my analysis",
        "Also Highlight the Stock names and company names in bold",
    ],
    debug_mode=True,
)

# Define request model
class QueryRequest(BaseModel):
    query: str

# Define response model
class QueryResponse(BaseModel):
    response: str

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def get_query_response(request: QueryRequest):
    try:
        # Get response from the agent
        response = agent.run(request.query, markdown=True).content
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
