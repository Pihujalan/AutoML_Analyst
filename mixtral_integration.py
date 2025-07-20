import requests
import json
import pandas as pd
import os
from datetime import datetime

class MixtralBusinessIntelligence:
    def __init__(self, together_api_key):
        self.api_key = together_api_key
        self.base_url = "https://api.together.xyz/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_business_report(self, analysis_results, model_metrics, plots_path="plots/"):
        """
        Generate comprehensive business intelligence report using Mixtral
        """
        
        # Collect all analysis data
        report_data = self._collect_analysis_data(analysis_results, model_metrics, plots_path)
        
        # Create the comprehensive prompt
        prompt = self._create_business_intelligence_prompt(report_data)
        
        # Call Mixtral API
        response = self._call_mixtral_api(prompt)
        
        return response
    
    def _collect_analysis_data(self, analysis_results, model_metrics, plots_path):
        """Collect all analysis data for the report"""
        
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_performance": model_metrics,
            "available_plots": [],
            "analysis_summary": {}
        }
        
        # Collect available plots
        if os.path.exists(plots_path):
            plot_files = os.listdir(plots_path)
            data["available_plots"] = [f for f in plot_files if f.endswith('.png')]
        
        # Collect analysis results
        if analysis_results:
            data["analysis_summary"] = analysis_results
        
        return data
    
    def _create_business_intelligence_prompt(self, report_data):
        """Create the comprehensive business intelligence prompt"""
        
        base_prompt = """You are a SENIOR BUSINESS INTELLIGENCE ANALYST with 15+ years of experience in data science, business strategy, and executive consulting. Your role is to transform raw data analysis into ACTIONABLE BUSINESS INTELLIGENCE that can replace a team of junior analysts and data engineers.

🎯 YOUR MISSION: Create a COMPREHENSIVE BUSINESS REPORT that enables executives to make data-driven decisions and run their business effectively.

📋 REPORT STRUCTURE (90% Data Analyst Replacement):

1. 🎯 EXECUTIVE SUMMARY (C-Level Ready)
   • 2-3 key business insights that matter to executives
   • Revenue impact and business implications
   • Risk assessment (High/Medium/Low)
   • Bottom line: What this means for the business

2. 📊 DATA QUALITY & BUSINESS RISK ASSESSMENT
   • Missing data: Business impact and data collection recommendations
   • Outliers: Fraud detection, data errors, or business opportunities?
   • Data completeness: Can we trust this data for decisions?
   • Data governance recommendations

3. 🔍 PATTERN ANALYSIS & BUSINESS INSIGHTS
   • What patterns reveal about customer behavior
   • Market opportunities and competitive advantages
   • Operational inefficiencies and optimization opportunities
   • Customer segmentation insights

4. 🤖 MODEL PERFORMANCE & BUSINESS VALUE
   • Model accuracy: Is it good enough for business decisions?
   • Overfitting/Underfitting: Business reliability assessment
   • Feature importance: What drives business outcomes?
   • Model fairness: Any bias that could hurt the business?
   • ROI potential: How much money can this save/make?

5. 📈 SEGMENT ANALYSIS & TARGETING
   • High-value customer segments
   • Underperforming segments and improvement opportunities
   • Geographic insights and expansion opportunities
   • Demographic targeting strategies

6. ⚠️ RISK ASSESSMENT & COMPLIANCE
   • Data privacy and regulatory risks
   • Model bias and fairness concerns
   • Operational risks and mitigation strategies
   • Financial risks and impact assessment

7. 💡 STRATEGIC RECOMMENDATIONS & ACTION PLAN
   • IMMEDIATE ACTIONS (Next 30 days):
     - Priority 1: [Specific action with expected outcome]
     - Priority 2: [Specific action with expected outcome]
     - Priority 3: [Specific action with expected outcome]

   • SHORT-TERM STRATEGY (3-6 months):
     - Strategic initiatives with resource requirements
     - Timeline and milestones
     - Success metrics and KPIs

   • LONG-TERM ROADMAP (1-2 years):
     - Strategic direction and investment opportunities
     - Technology and process improvements
     - Market expansion and growth strategies

8. 📊 SUCCESS METRICS & MONITORING
   • Key Performance Indicators (KPIs) to track
   • Dashboard recommendations
   • Alert thresholds and monitoring systems
   • ROI measurement framework

9. 🔧 IMPLEMENTATION GUIDANCE
   • Technical requirements and resources needed
   • Team structure and skill requirements
   • Timeline and project management
   • Budget considerations and cost-benefit analysis

10. 🎯 BUSINESS INTELLIGENCE DASHBOARD RECOMMENDATIONS
    • Key metrics to display for executives
    • Real-time monitoring capabilities
    • Automated reporting and alerting
    • User access and security considerations

📌 WRITING STYLE:
• Write like a SENIOR BUSINESS CONSULTANT, not an AI
• Use BUSINESS LANGUAGE, not technical jargon
• Provide SPECIFIC, ACTIONABLE recommendations
• Include EXPECTED OUTCOMES and ROI estimates
• Use BULLET POINTS and CLEAR HEADERS
• Make it EASY for non-technical executives to understand
• Focus on BUSINESS VALUE, not just data insights

🎯 FINAL GOAL: This report should enable executives to make informed business decisions and run their operations more effectively. Think like a McKinsey consultant delivering to a Fortune 500 CEO.

ANALYSIS DATA TO INTERPRET:
"""
        
        # Add the analysis data to the prompt
        data_summary = f"""
MODEL PERFORMANCE METRICS:
{json.dumps(report_data['model_performance'], indent=2)}

AVAILABLE VISUALIZATIONS:
{', '.join(report_data['available_plots'])}

ANALYSIS TIMESTAMP:
{report_data['timestamp']}

Please analyze this data and create a comprehensive business intelligence report following the structure above.
"""
        
        return base_prompt + data_summary
    
    def _call_mixtral_api(self, prompt):
        """Call the Together AI Mixtral API"""
        
        payload = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling Mixtral API: {str(e)}"
    
    def save_report(self, report_content, filename="business_intelligence_report.md"):
        """Save the generated report to a file"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ Business Intelligence Report saved to: {filename}")
        return filename

# Integration function to be called from your existing pipeline
def integrate_mixtral_business_intelligence(together_api_key, analysis_results=None, model_metrics=None):
    """
    Main integration function to add business intelligence to your AutoML pipeline
    
    Args:
        together_api_key: Your Together AI API key
        analysis_results: Results from your pattern analysis
        model_metrics: Model performance metrics
    """
    
    # Initialize the business intelligence generator
    bi_generator = MixtralBusinessIntelligence(together_api_key)
    
    # Generate the comprehensive business report
    print("🤖 Generating Business Intelligence Report with Mixtral...")
    report = bi_generator.generate_business_report(analysis_results, model_metrics)
    
    # Save the report
    filename = bi_generator.save_report(report)
    
    return report, filename

# Example usage
if __name__ == "__main__":
    # Load your existing model metrics
    try:
        with open("final_model_summary.json", "r") as f:
            model_metrics = json.load(f)
    except:
        model_metrics = {"model_used": "unknown", "test_metrics": {}}
    
    # Your Together AI API key
    TOGETHER_API_KEY = "your_together_api_key_here"
    
    # Generate business intelligence report
    report, filename = integrate_mixtral_business_intelligence(
        together_api_key=TOGETHER_API_KEY,
        model_metrics=model_metrics
    )
    
    print(f"\n📊 Business Intelligence Report Generated!")
    print(f"📁 Saved to: {filename}")
    print(f"📄 Report length: {len(report)} characters") 