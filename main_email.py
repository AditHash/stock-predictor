from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch
from phi.tools.yfinance import YFinanceTools
from phi.tools.email import EmailTools  # Import EmailTools for sending emails
from phi.model.aws.claude import Claude  # For AWS Bedrock Claude model
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Ensure AWS credentials are set via environment variables or CLI configuration
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")  # Default to us-east-1

# Email configuration
receiver_email = os.getenv("RECEIVER_EMAIL")  # Receiver email address
sender_email = os.getenv("SENDER_EMAIL")  # Sender email address
sender_name = os.getenv("SENDER_NAME", "Investment Advisor")  # Sender name
sender_passkey = os.getenv("SENDER_PASSKEY")  # Sender passkey or app password

# Initialize FastAPI app
app = FastAPI()

# Create an agent with combined capabilities, using Claude from AWS Bedrock
agent = Agent(
    model=Claude(id="anthropic.claude-3-5-sonnet-20240620-v1:0"),  # Claude model from AWS Bedrock
    tools=[
        DuckDuckGo(),
        GoogleSearch(),
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True),
        EmailTools(  # Configure the email tool
            receiver_email=receiver_email,
            sender_email=sender_email,
            sender_name=sender_name,
            sender_passkey=sender_passkey,
        ),
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
        # Generate the response in markdown format
        markdown_response = await agent.run(request.query, markdown=True)
        markdown_content = markdown_response.content  # Extract the content

        # Convert the markdown response into plain text for email
        plain_text_content = markdown_content.replace("**", "")  # Simplistic conversion for example

        # Send the plain text response via email
        email_query = (
            f"send an email to {receiver_email} with the following content:\n\n{plain_text_content}"
        )
        await agent.run(email_query)  # Send email asynchronously

        # Return the markdown response for the API
        return QueryResponse(response=markdown_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7090)
