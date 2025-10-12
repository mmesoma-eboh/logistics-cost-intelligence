"""
Logistics Cost Analysis - Visualization Generator - FIXED VERSION
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

print("=== LOGISTICS COST ANALYSIS - VISUALIZATION GENERATOR ===")
print("✅ Libraries imported successfully!")

# Create visuals directory if it doesn't exist
os.makedirs('visuals', exist_ok=True)

# Set professional styling
plt.style.use('default')
sns.set_palette("husl")

# =============================================================================
# CREATE SAMPLE DATA
# =============================================================================
print("\n📊 Creating sample business data for visualization...")

np.random.seed(42)
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

print("✅ Sample data created successfully!")

# =============================================================================
# 1. EXECUTIVE DASHBOARD
# =============================================================================
print("\n📈 Creating Executive Dashboard...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Logistics Cost Intelligence: $200K+ Annual Savings Opportunity', 
             fontsize=16, fontweight='bold')

# Plot 1: Two-Stream Cost Analysis
streams = ['Core Freight\n(Predictable)', 'Volatile Costs\n(Service Fees)']
spends = [285435, 133000]
colors = ['#2E8B57', '#FF6B6B']

bars = ax1.bar(streams, spends, color=colors, alpha=0.8, edgecolor='black')
ax1.set_title('Two-Stream Cost Analysis', fontweight='bold')
ax1.set_ylabel('Total Spend ($)')

for bar, spend in zip(bars, spends):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
            f'${spend/1000:.0f}K', ha='center', va='bottom', fontweight='bold')

# Plot 2: NYC-FRA Consolidation Opportunity
scenarios = ['Current\nPerformance', 'With Consolidation\n(Target)']
costs = [109.33, 43.73]

bars = ax2.bar(scenarios, costs, color=['#FF6B6B', '#2E8B57'], alpha=0.8)
ax2.set_title('NYC-FRA: 60%+ Cost Reduction Opportunity', fontweight='bold')
ax2.set_ylabel('Cost per KG ($)')

# Plot 3: Business Case Summary
opportunities = ['Consolidation\nSavings', 'Budgeting\nEfficiency', 'Total\nOpportunity']
values = [75000, 50000, 225000]

bars = ax3.bar(opportunities, values, color=['#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
ax3.set_title('$200K+ Annual Savings Opportunity', fontweight='bold')
ax3.set_ylabel('Annual Value ($)')

for bar, value in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
            f'${value/1000:.0f}K', ha='center', va='bottom', fontweight='bold')

# Plot 4: Simple Roadmap
roadmap_data = ['Immediate Pilot\n(0-3 months)', 'Strategic Expansion\n(3-12 months)', 'Transformation\n(FY2026)']
ax4.bar(roadmap_data, [1, 1, 1], color=['#FF9999', '#FFB366', '#99FF99'], alpha=0.7)
ax4.set_title('Implementation Roadmap', fontweight='bold')
ax4.set_ylabel('Phase')

plt.tight_layout()
plt.savefig('visuals/executive_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Executive dashboard saved: visuals/executive_dashboard.png")

# =============================================================================
# 2. COST DISTRIBUTION ANALYSIS
# =============================================================================
print("\n📊 Creating Cost Distribution Analysis...")

plt.figure(figsize=(12, 8))

# Cost by Lane
lane_cost = modeling_data.groupby('lane')['cleaned_cost_per_kg'].mean()
bars = plt.bar(lane_cost.index, lane_cost.values, alpha=0.7, edgecolor='black')
plt.title('Average Cost per Kg by Lane', fontsize=14, fontweight='bold')
plt.ylabel('Cost per Kg ($)', fontweight='bold')
plt.xticks(rotation=45)

for bar, cost in zip(bars, lane_cost.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'${cost:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('visuals/cost_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Cost distribution analysis saved: visuals/cost_distribution_analysis.png")

# =============================================================================
# 3. BUSINESS DRIVER IMPORTANCE
# =============================================================================
print("\n🎯 Creating Business Driver Importance Chart...")

features = ['Service Level', 'Monthly Avg Cost', 'Lane', 'Shipment Weight', 'Volatility']
importance = [0.254, 0.198, 0.167, 0.145, 0.123]

plt.figure(figsize=(10, 6))
bars = plt.barh(features, importance, color='skyblue', alpha=0.7, edgecolor='navy')
plt.title('Business Driver Importance', fontsize=14, fontweight='bold')
plt.xlabel('Feature Importance Score')
plt.grid(axis='x', alpha=0.3)

for bar, imp in zip(bars, importance):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{imp:.3f}', va='center', ha='left', fontweight='bold')

plt.tight_layout()
plt.savefig('visuals/business_driver_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Business driver importance saved: visuals/business_driver_importance.png")

# =============================================================================
# 4. TWO-STREAM COST ANALYSIS
# =============================================================================
print("\n💰 Creating Two-Stream Cost Analysis...")

plt.figure(figsize=(8, 8))
streams = ['Core Freight\n(Predictable, 68%)', 'Volatile Costs\n(Service Fees, 32%)']
spends = [68, 32]
colors = ['#2E8B57', '#FF6B6B']

plt.pie(spends, labels=streams, colors=colors, autopct='%1.0f%%', startangle=90)
plt.title('Two-Stream Cost Analysis', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('visuals/two_stream_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Two-stream analysis saved: visuals/two_stream_analysis.png")

print("\n🎯 VISUALIZATION GENERATION COMPLETED!")
print("📁 Generated 4 professional business visuals in visuals/ folder")
