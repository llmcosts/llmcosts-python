# Client Tracking & Context Data Guide

One of LLMCosts' most powerful features is seamless **client-level cost tracking** and **custom context data** for every API call. Perfect for agencies, SaaS platforms, and any multi-tenant application where you need to track costs per client, user, project, or department.

**🔒 Privacy-First**: Even with rich context tracking, LLMCosts NEVER sees your API keys, requests, or responses. Only the usage metadata and context data you choose to include is transmitted.

**🔄 Universal**: Works with ANY LLM provider - OpenAI, Anthropic, Google, AWS Bedrock, and more.

## 🎯 Key Benefits

- **📊 Per-Client Cost Tracking**: Automatically track LLM costs per customer, user, or tenant
- **🏷️ Rich Context Data**: Add any metadata - project names, departments, user IDs, session data
- **💰 Billing Integration**: Perfect for client billing, quota management, and cost allocation
- **📈 Analytics**: Detailed usage analytics with custom dimensions
- **🔄 Dynamic**: Change client/context mid-session without restarting

## 🏢 Client-Level Tracking

### Customer Key Tracking

Use `client_customer_key` to track usage per customer, tenant, or billing entity:

```python
from llmcosts import LLMTrackingProxy, Provider
import openai

client = openai.OpenAI(api_key="your-key")
tracked_client = LLMTrackingProxy(
    client,
    provider=Provider.OPENAI,
    client_customer_key="customer_acme_corp"  # Your customer identifier
)

# All API calls automatically include this customer key
response = tracked_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
# → Logged with: {"client_customer_key": "customer_acme_corp", ...}
```

### Dynamic Customer Switching

Perfect for multi-tenant applications:

```python
# Start with one customer
tracked_client.client_customer_key = "customer_123"
response1 = tracked_client.chat.completions.create(...)

# Switch to another customer mid-session
tracked_client.client_customer_key = "customer_456" 
response2 = tracked_client.chat.completions.create(...)

# Each call is properly attributed to the correct customer
```

## 🏷️ Custom Context Data

### Rich Metadata Tracking

Add any context data you need for analytics and cost allocation:

```python
tracked_client = LLMTrackingProxy(
    client,
    provider=Provider.OPENAI,
    client_customer_key="acme_corp",
    context={
        "user_id": "user_789",
        "session_id": "session_abc123",
        "project": "chatbot_v2",
        "department": "customer_support",
        "environment": "production",
        "feature": "chat_completion",
        "cost_center": "support_team",
        "app_version": "2.1.4"
    }
)

response = tracked_client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": "Help request"}]
)
# → All context data included in usage logs
```

### Dynamic Context Updates

Update context data throughout your application lifecycle:

```python
# Initial context
tracked_client.context = {
    "user_id": "user_123",
    "session_id": "session_new"
}

# Add more context as user navigates
tracked_client.context.update({
    "current_page": "dashboard",
    "user_tier": "premium"
})

# Switch to different feature
tracked_client.context.update({
    "feature": "document_analysis",
    "document_id": "doc_456"
})

# Each API call includes the current context state
```

## 🏗️ Multi-Tenant Applications

### SaaS Platform Example

```python
import time
from llmcosts import LLMTrackingProxy, Provider
import openai

class LLMService:
    def __init__(self, openai_api_key, llmcosts_api_key):
        self.base_client = openai.OpenAI(api_key=openai_api_key)
        self.tracked_client = LLMTrackingProxy(
            self.base_client,
            provider=Provider.OPENAI,
            api_key=llmcosts_api_key
        )
    
    def process_for_customer(self, customer_id, user_id, project_id, prompt):
        # Set customer for billing
        self.tracked_client.client_customer_key = customer_id
        
        # Rich context for analytics
        self.tracked_client.context = {
            "user_id": user_id,
            "project_id": project_id,
            "timestamp": time.time(),
            "service": "text_generation",
            "billing_tier": self._get_customer_tier(customer_id)
        }
        
        return self.tracked_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
    
    def _get_customer_tier(self, customer_id):
        # Your business logic here
        return "enterprise" if customer_id.startswith("ent_") else "standard"

# Usage
service = LLMService("openai-key", "llmcosts-key")

# Customer A - Enterprise
response1 = service.process_for_customer(
    "ent_acme_corp", "user_123", "proj_website", "Write a blog post"
)

# Customer B - Standard  
response2 = service.process_for_customer(
    "std_startup_inc", "user_456", "proj_app", "Generate product descriptions"
)
```

### Agency/Consultancy Example

```python
class AgencyLLMService:
    def __init__(self, openai_api_key, llmcosts_api_key):
        self.base_client = openai.OpenAI(api_key=openai_api_key)
        self.tracked_client = LLMTrackingProxy(
            self.base_client,
            provider=Provider.OPENAI,
            api_key=llmcosts_api_key
        )
    
    def work_for_client(self, client_name, project_name, team_member, task_type, content):
        # Client billing tracking
        self.tracked_client.client_customer_key = f"client_{client_name.lower()}"
        
        # Project and team context
        self.tracked_client.context = {
            "client": client_name,
            "project": project_name,
            "team_member": team_member,
            "task_type": task_type,
            "billable": True,
            "department": "creative",
            "timestamp": time.time()
        }
        
        return self.tracked_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}]
        )

# Usage  
agency = AgencyLLMService("openai-key", "llmcosts-key")

# Work for different clients
agency.work_for_client(
    "Acme Corp", "Brand Refresh", "sarah@agency.com", 
    "copywriting", "Write website copy"
)

agency.work_for_client(
    "Startup Inc", "Product Launch", "john@agency.com",
    "content_strategy", "Create blog content plan" 
)
```

## 📊 Usage Data Output

With client tracking and context data, your usage logs become rich with actionable information:

```json
{
  "usage": {
    "completion_tokens": 150,
    "prompt_tokens": 50,
    "total_tokens": 200
  },
  "model_id": "gpt-4o-mini",
  "response_id": "chatcmpl-123abc",
  "timestamp": "2024-01-15T10:30:00Z",
  "provider": "openai",
  "client_customer_key": "customer_acme_corp",
  "context": {
    "user_id": "user_789",
    "project": "chatbot_v2", 
    "department": "customer_support",
    "environment": "production",
    "cost_center": "support_team"
  }
}
```

## 🎯 Use Cases

### Cost Allocation

- **Per-Customer Billing**: Track exact costs per customer for billing
- **Department Budgets**: Allocate AI costs to specific departments
- **Project Tracking**: Monitor costs per project or initiative
- **User Quotas**: Set and track per-user usage limits

### Analytics & Optimization

- **Usage Patterns**: Understand how different customers use AI
- **Cost Optimization**: Identify high-cost users or use cases
- **Performance**: Track model performance across different contexts
- **Scaling**: Plan capacity based on customer growth

### Business Intelligence

- **Revenue Attribution**: Link AI costs to revenue-generating activities
- **Customer Insights**: Understand customer AI consumption patterns
- **Efficiency Metrics**: Track AI efficiency across teams/projects
- **ROI Analysis**: Measure AI ROI per customer or use case

## 🔧 Advanced Patterns

### Context Inheritance

```python
# Base context for all operations
tracked_client.context = {
    "app_version": "2.1.0",
    "environment": "production",
    "region": "us-east-1"
}

# Function-specific context (inherits base)
def analyze_document(doc_id, user_id):
    # Temporarily add specific context
    original_context = tracked_client.context.copy()
    tracked_client.context.update({
        "feature": "document_analysis",
        "document_id": doc_id,
        "user_id": user_id
    })
    
    try:
        response = tracked_client.chat.completions.create(...)
        return response
    finally:
        # Restore original context
        tracked_client.context = original_context
```

### Middleware Pattern

```python
class ContextMiddleware:
    def __init__(self, tracked_client):
        self.tracked_client = tracked_client
    
    def with_context(self, **context_data):
        """Context manager for temporary context"""
        original_context = self.tracked_client.context.copy() if self.tracked_client.context else {}
        
        class ContextManager:
            def __enter__(self):
                self.tracked_client.context = {**original_context, **context_data}
                return self.tracked_client
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.tracked_client.context = original_context
        
        return ContextManager()

# Usage
middleware = ContextMiddleware(tracked_client)

with middleware.with_context(user_id="123", feature="chat"):
    response = tracked_client.chat.completions.create(...)
    # Automatic context cleanup
```

## 📈 Analytics Integration

The rich context data integrates perfectly with your analytics stack:

```python
# Example: Send to your analytics platform
def analytics_callback(response):
    """Custom callback to send data to analytics"""
    if hasattr(response, 'usage') and tracked_client.context:
        analytics_event = {
            "event": "llm_usage",
            "customer_id": tracked_client.client_customer_key,
            "tokens": response.usage.total_tokens,
            "model": response.model,
            "cost_estimate": calculate_cost(response.usage.total_tokens),
            **tracked_client.context  # Include all context data
        }
        
        # Send to your analytics platform
        analytics.track(analytics_event)

tracked_client.response_callback = analytics_callback
```

## Next Steps

- **[Configuration](configuration.md)** - Advanced configuration options
- **[Providers](providers.md)** - Provider-specific integration guides  
- **[Pricing](pricing.md)** - Cost calculation and model discovery
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions 