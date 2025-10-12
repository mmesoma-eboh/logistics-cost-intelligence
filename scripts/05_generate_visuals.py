"""
Logistics Cost Analysis - Visualization Generator
Creates professional business visuals for portfolio and presentations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

print("=== LOGISTICS COST ANALYSIS - VISUALIZATION GENERATOR ===")
print("✅ Libraries imported successfully!")

# Set professional styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# LOAD BUSINESS DATA
# =============================================================================
print("\n📊 Loading business data for visualization...")

try:
    modeling_data = pd.read_csv('../data/modeling_dataset.csv')
    business_impact = pd.read_csv('../data/business_impact_summary.csv')
    two_stream = pd.read_csv('../data/two_stream_analysis.csv')
    feature_importance = pd.read_csv('../data/feature_importance.csv')
    
    print("✅ Data loaded successfully!")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("💡 Creating sample data for demonstration...")
    # Create sample data if files don't exist
    modeling_data = pd.DataFrame({
        'lane': ['NYC-FRA', 'SHANGHAI-CHICAGO', 'LONDON-SINGAPORE'] * 30,
        'cleaned_cost_per_kg': np.concatenate([
            np.random.normal(40, 15, 30),  # NYC-FRA - higher volatility
            np.random.normal(22, 8, 30),   # SHANGHAI-CHICAGO
            np.random.normal(24, 9, 30)    # LONDON-SINGAPORE
        ]),
        'service_level': np.random.choice(['Standard', 'Express', 'Priority'], 90),
        'is_nyc_fra_small': np.random.choice([0, 1], 90, p=[0.7, 0.3])
    })

# =============================================================================
# 1. EXECUTIVE DASHBOARD - BUSINESS IMPACT SUMMARY
# =============================================================================
print("\n📈 Creating Executive Dashboard...")

fig = plt.figure(figsize=(20, 12))
fig.suptitle('Logistics Cost Intelligence: $200k+ Annual Savings Opportunity', 
             fontsize=18, fontweight='bold', y=0.95)

# Create grid layout
gs = plt.GridSpec(3, 3, figure=fig)

# Plot 1: Two-Stream Cost Analysis
ax1 = fig.add_subplot(gs[0, 0])
streams = ['Core Freight\n(Predictable)', 'Volatile Costs\n(Service Fees)']
spends = [285435, 133000]  # From two-stream analysis
colors = ['#2E8B57', '#FF6B6B']

bars = ax1.bar(streams, spends, color=colors, alpha=0.8, edgecolor='black')
ax1.set_title('Two-Stream Cost Analysis\n(Proactive Budgeting Foundation)', 
              fontweight='bold', fontsize=12)
ax1.set_ylabel('Total Spend ($)', fontweight='bold')

# Add value labels
for bar, spend in zip(bars, spends):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
            f'${spend/1000:.0f}K', ha='center', va='bottom', 
            fontweight='bold', fontsize=11)

# Plot 2: NYC-FRA Consolidation Opportunity
ax2 = fig.add_subplot(gs[0, 1])
scenarios = ['Current\nPerformance', 'With Consolidation\n(Target)']
costs = [109.33, 43.73]  # 60% reduction

bars = ax2.bar(scenarios, costs, color=['#FF6B6B', '#2E8B57'], 
               alpha=0.8, edgecolor='black')
ax2.set_title('NYC-FRA Lane: 60%+ Cost Reduction Opportunity', 
              fontweight='bold', fontsize=12)
ax2.set_ylabel('Cost per KG ($)', fontweight='bold')

# Add savings annotation
savings = 75000
ax2.text(0.5, max(costs) * 0.8, f'Potential Savings:\n${savings/1000:.0f}K/year', 
        ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# Plot 3: Business Case Summary
ax3 = fig.add_subplot(gs[0, 2])
opportunities = ['Consolidation\nSavings', 'Budgeting\nEfficiency', 'Total\nOpportunity']
values = [75000, 50000, 225000]

bars = ax3.bar(opportunities, values, color=['#4ECDC4', '#45B7D1', '#96CEB4'], 
               alpha=0.8, edgecolor='black')
ax3.set_title('$200k+ Annual Savings Opportunity', 
              fontweight='bold', fontsize=12)
ax3.set_ylabel('Annual Value ($)', fontweight='bold')

# Add value labels
for bar, value in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
            f'${value/1000:.0f}K', ha='center', va='bottom', 
            fontweight='bold', fontsize=11)

# Plot 4: Implementation Roadmap
ax4 = fig.add_subplot(gs[1, :])
roadmap_data = [
    ('Q1 2024', 'NYC-FRA Consolidation Pilot', 'Immediate', '$50-75K Savings'),
    ('Q2 2024', 'Expand to Other Lanes', '3-6 months', 'Additional $50K'),
    ('FY2026', 'Proactive Budgeting Model', 'Planning Cycle', 'Variance Elimination'),
    ('Ongoing', 'Profit Center Transformation', 'Continuous', 'Competitive Advantage')
]

# Create timeline
y_pos = range(len(roadmap_data))
colors = ['#FF9999', '#FFB366', '#99FF99', '#66B2FF']

for i, (timing, initiative, duration, impact) in enumerate(roadmap_data):
    rect = plt.Rectangle((0.1, i), 0.8, 0.8, fill=True, color=colors[i], 
                        alpha=0.7, transform=ax4.transData)
    ax4.add_patch(rect)
    ax4.text(0.15, i + 0.6, initiative, va='center', ha='left', 
             fontweight='bold', fontsize=10, transform=ax4.transData)
    ax4.text(0.15, i + 0.3, impact, va='center', ha='left', 
             fontsize=9, transform=ax4.transData)

ax4.set_xlim(0, 1)
ax4.set_ylim(-0.5, len(roadmap_data) - 0.5)
ax4.set_yticks([i + 0.4 for i in range(len(roadmap_data))])
ax4.set_yticklabels([data[0] for data in roadmap_data])
ax4.set_title('Implementation Roadmap & Timeline', fontweight='bold', fontsize=14)
ax4.axis('off')

# Plot 5: Key Performance Indicators
ax5 = fig.add_subplot(gs[2, :])
kpis = [
    ('98.3%', 'Forecast Accuracy\n(Core Freight)'),
    ('60%+', 'Cost Reduction\n(NYC-FRA Lane)'),
    ('$200K+', 'Annual Savings\nPotential'),
    ('FY2026', 'Budget Model\nImplementation')
]

# Create KPI boxes
for i, (value, label) in enumerate(kpis):
    rect = plt.Rectangle((i * 0.25, 0), 0.2, 1, fill=True, 
                        color='lightblue', alpha=0.7, transform=ax5.transData)
    ax5.add_patch(rect)
    ax5.text(i * 0.25 + 0.1, 0.7, value, ha='center', va='center', 
             fontsize=16, fontweight='bold', transform=ax5.transData)
    ax5.text(i * 0.25 + 0.1, 0.3, label, ha='center', va='center', 
             fontsize=10, transform=ax5.transData)

ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis('off')
ax5.set_title('Key Performance Indicators', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('../visuals/executive_dashboard.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("✅ Executive dashboard saved: ../visuals/executive_dashboard.png")

# =============================================================================
# 2. COST DISTRIBUTION ANALYSIS
# =============================================================================
print("\n📊 Creating Cost Distribution Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Logistics Cost Distribution Analysis - Business Insights', 
             fontsize=16, fontweight='bold')

# Plot 1: Cost by Lane with NYC-FRA focus
lane_cost = modeling_data.groupby('lane')['cleaned_cost_per_kg'].mean().sort_values(ascending=False)
bars = axes[0,0].bar(lane_cost.index, lane_cost.values, alpha=0.7, edgecolor='black')
axes[0,0].set_title('Average Cost per Kg by Lane\n(NYC-FRA Optimization Opportunity)', 
                    fontweight='bold')
axes[0,0].set_ylabel('Cost per Kg ($)', fontweight='bold')
axes[0,0].tick_params(axis='x', rotation=45)

# Highlight NYC-FRA
nyc_fra_cost = lane_cost['NYC-FRA'] if 'NYC-FRA' in lane_cost.index else lane_cost.iloc[0]
axes[0,0].axhline(y=nyc_fra_cost, color='red', linestyle='--', alpha=0.7, 
                 label=f'NYC-FRA: ${nyc_fra_cost:.1f}')
axes[0,0].legend()

# Plot 2: Cost Distribution
axes[0,1].hist(modeling_data['cleaned_cost_per_kg'], bins=20, alpha=0.7, 
               color='green', edgecolor='black')
axes[0,1].axvline(modeling_data['cleaned_cost_per_kg'].mean(), color='red', 
                 linestyle='--', label=f'Mean: ${modeling_data["cleaned_cost_per_kg"].mean():.1f}')
axes[0,1].set_title('Cleaned Cost Distribution\n(Standard Freight Only)', fontweight='bold')
axes[0,1].set_xlabel('Cost per Kg ($)', fontweight='bold')
axes[0,1].set_ylabel('Frequency', fontweight='bold')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: NYC-FRA Small vs Large Shipments
if 'is_nyc_fra_small' in modeling_data.columns:
    nyc_fra_data = modeling_data[modeling_data['lane'] == 'NYC-FRA'] if 'lane' in modeling_data.columns else modeling_data
    small_shipments = nyc_fra_data[nyc_fra_data['is_nyc_fra_small'] == 1]['cleaned_cost_per_kg'].mean()
    large_shipments = nyc_fra_data[nyc_fra_data['is_nyc_fra_small'] == 0]['cleaned_cost_per_kg'].mean()
    
    scenarios = ['Small Shipments\n(<20kg)', 'Large Shipments\n(≥20kg)']
    costs = [small_shipments, large_shipments]
    
    bars = axes[1,0].bar(scenarios, costs, color=['red', 'green'], alpha=0.7, edgecolor='black')
    axes[1,0].set_title('NYC-FRA: Consolidation Impact Analysis', fontweight='bold')
    axes[1,0].set_ylabel('Average Cost per Kg ($)', fontweight='bold')
    
    # Add cost labels
    for bar, cost in zip(bars, costs):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                      f'${cost:.1f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Service Level Impact
if 'service_level' in modeling_data.columns:
    service_cost = modeling_data.groupby('service_level')['cleaned_cost_per_kg'].mean().sort_values(ascending=False)
    bars = axes[1,1].bar(service_cost.index, service_cost.values, 
                        color=['red', 'orange', 'green'], alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Service Level Cost Premiums', fontweight='bold')
    axes[1,1].set_ylabel('Average Cost per Kg ($)', fontweight='bold')
    
    # Add premium percentages
    if len(service_cost) > 1:
        standard_cost = service_cost.iloc[-1]  # Assuming Standard is lowest
        for i, (service, cost) in enumerate(service_cost.items()):
            if service != 'Standard':
                premium = ((cost - standard_cost) / standard_cost) * 100
                axes[1,1].text(i, cost + 2, f'+{premium:.0f}%', 
                              ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('../visuals/cost_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Cost distribution analysis saved: ../visuals/cost_distribution_analysis.png")

# =============================================================================
# 3. BUSINESS DRIVER IMPORTANCE
# =============================================================================
print("\n🎯 Creating Business Driver Importance Chart...")

plt.figure(figsize=(12, 8))

# Use feature importance data or create sample
if not feature_importance.empty:
    features = feature_importance.sort_values('importance_score', ascending=True)
else:
    # Sample feature importance
    features = pd.DataFrame({
        'feature': ['Service Level', 'Monthly Avg Cost', 'Lane', 'Shipment Weight', 
                   'Volatility Ratio', 'Carrier', 'NYC-FRA Small'],
        'importance_score': [0.254, 0.198, 0.167, 0.145, 0.123, 0.087, 0.026]
    })

bars = plt.barh(features['feature'], features['importance_score'], 
                color='skyblue', alpha=0.7, edgecolor='navy')

plt.title('Business Driver Importance - Predictive Model Insights', 
          fontsize=16, fontweight='bold')
plt.xlabel('Feature Importance Score (Higher = More Predictive)', fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# Add value labels
for bar, importance in zip(bars, features['importance_score']):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{importance:.3f}', va='center', ha='left', fontweight='bold')

plt.tight_layout()
plt.savefig('../visuals/business_driver_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Business driver importance saved: ../visuals/business_driver_importance.png")

# =============================================================================
# 4. TWO-STREAM COST ANALYSIS VISUAL
# =============================================================================
print("\n💰 Creating Two-Stream Cost Analysis...")

plt.figure(figsize=(10, 8))

# Two-stream data
streams = ['Core Freight\n(Predictable, 68%)', 'Volatile Costs\n(Service Fees, 32%)']
spends = [285435, 133000]
colors = ['#2E8B57', '#FF6B6B']

# Create pie chart with business context
wedges, texts, autotexts = plt.pie(spends, labels=streams, colors=colors, autopct='%1.1f%%',
                                  startangle=90, explode=(0.05, 0))

# Enhance labels
for text in texts:
    text.set_fontweight('bold')
    text.set_fontsize(11)
for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_color('white')

plt.title('Two-Stream Cost Analysis\n(Separating Predictable vs Volatile Costs)', 
          fontsize=14, fontweight='bold')

# Add business impact annotation
plt.annotate('98.3% Forecasting Accuracy\nAchievable for Core Freight', 
             xy=(0.5, 0.5), xytext=(0.8, 0.8),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'),
             fontweight='bold', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

plt.tight_layout()
plt.savefig('../visuals/two_stream_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Two-stream analysis saved: ../visuals/two_stream_analysis.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n🎯 VISUALIZATION GENERATION COMPLETE!")
print("=" * 50)
print("📁 Generated Files:")
print("   - ../visuals/executive_dashboard.png")
print("   - ../visuals/cost_distribution_analysis.png") 
print("   - ../visuals/business_driver_importance.png")
print("   - ../visuals/two_stream_analysis.png")
