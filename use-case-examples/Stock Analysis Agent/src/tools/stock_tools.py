import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import os

def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a given ticker."""
    try:
        stock = yf.Ticker(ticker)
        # Try to get fast info first, then fallback to history
        if hasattr(stock, 'fast_info') and 'last_price' in stock.fast_info:
            price = stock.fast_info['last_price']
        else:
            history = stock.history(period="1d")
            if history.empty:
                return f"Could not find price data for {ticker}."
            price = history['Close'].iloc[-1]
            
        return f"The current price of {ticker.upper()} is ${price:.2f}"
    except Exception as e:
        return f"Error fetching price for {ticker}: {str(e)}"

def get_company_info(ticker: str) -> str:
    """Get summary information about a company."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        name = info.get('longName', ticker.upper())
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        summary = info.get('longBusinessSummary', 'No summary available.')
        
        # Truncate summary if too long
        if len(summary) > 500:
            summary = summary[:500] + "..."
            
        return f"**{name}**\nSector: {sector}\nIndustry: {industry}\n\n{summary}"
    except Exception as e:
        return f"Error fetching info for {ticker}: {str(e)}"

def plot_stock_history(ticker: str, period: str = "1y") -> str:
    """
    Plot the stock history for a given ticker and period.
    Period options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    Returns JSON representation of the plotly figure and events summary.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return f"No historical data found for {ticker} with period {period}."
        
        # Calculate daily percentage changes
        hist['Pct_Change'] = hist['Close'].pct_change() * 100
        
        # Identify significant movements (>3% change)
        significant_moves = hist[abs(hist['Pct_Change']) > 3].copy()
        
        # Create interactive plotly chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#00cc96', width=2),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # Add markers for significant movements
        if not significant_moves.empty:
            # Separate spikes (positive) and dips (negative)
            spikes = significant_moves[significant_moves['Pct_Change'] > 0]
            dips = significant_moves[significant_moves['Pct_Change'] < 0]
            
            if not spikes.empty:
                fig.add_trace(go.Scatter(
                    x=spikes.index,
                    y=spikes['Close'],
                    mode='markers',
                    name='Major Spikes',
                    marker=dict(color='lime', size=10, symbol='triangle-up'),
                    hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: $%{y:.2f}<br><b>Change</b>: +%{customdata:.1f}%<extra></extra>',
                    customdata=spikes['Pct_Change']
                ))
            
            if not dips.empty:
                fig.add_trace(go.Scatter(
                    x=dips.index,
                    y=dips['Close'],
                    mode='markers',
                    name='Major Dips',
                    marker=dict(color='red', size=10, symbol='triangle-down'),
                    hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: $%{y:.2f}<br><b>Change</b>: %{customdata:.1f}%<extra></extra>',
                    customdata=dips['Pct_Change']
                ))
        
        fig.update_layout(
            title=f"{ticker.upper()} Stock Price - {period}",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            template='plotly_dark',
            height=500,
            showlegend=True
        )
        
        # Save figure as JSON for transfer
        import json
        fig_json = fig.to_json()
        
        # Create summary of significant events
        events_summary = ""
        if not significant_moves.empty:
            events_summary = "\n\n**Significant Price Movements Detected:**\n"
            top_moves = significant_moves.nlargest(5, 'Pct_Change', keep='all')
            bottom_moves = significant_moves.nsmallest(5, 'Pct_Change', keep='all')
            all_moves = pd.concat([top_moves, bottom_moves]).sort_index(ascending=False).head(10)
            
            for date, row in all_moves.iterrows():
                change = row['Pct_Change']
                marker = "▲" if change > 0 else "▼"
                sign = "+" if change > 0 else ""
                events_summary += f"\n- **{date.strftime('%Y-%m-%d')}** {marker} {sign}{change:.1f}% (${row['Close']:.2f})"
            
            events_summary += "\n\n*Tip: Search news for these dates to understand what caused these movements.*"
        
        return f"PLOTLY_CHART:{fig_json}{events_summary}"
    except Exception as e:
        return f"Error plotting history for {ticker}: {str(e)}"

def simulate_investment(ticker: str, amount: float, period: str = "1y") -> str:
    """
    Simulate what would happen if you invested a certain amount in a stock.
    
    Args:
        ticker: Stock ticker symbol
        amount: Investment amount in dollars
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
        String with simulation results and chart path
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return f"No historical data found for {ticker} with period {period}."
        
        # Get starting and ending prices
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        
        # Calculate shares and returns
        shares = amount / start_price
        end_value = shares * end_price
        profit = end_value - amount
        profit_pct = (profit / amount) * 100
        
        # Create visualization
        fig = go.Figure()
        
        # Calculate portfolio value over time
        portfolio_value = (hist['Close'] / start_price) * amount
        
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=portfolio_value,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00cc96', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 204, 150, 0.2)',
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Value</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # Add initial investment line
        fig.add_hline(
            y=amount,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Initial: ${amount:.2f}",
            annotation_position="right"
        )
        
        fig.update_layout(
            title=f"What If: ${amount:.0f} invested in {ticker.upper()} - {period}",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            template='plotly_dark',
            height=500,
            showlegend=True
        )
        
        # Save figure as JSON for transfer
        import json
        fig_json = fig.to_json()
        
        # Format response
        profit_label = "Profit" if profit >= 0 else "Loss"
        return_label = "Return" if profit >= 0 else "Return"
        
        result = f"""**Investment Simulation Results**

**Initial Investment:** ${amount:.2f}
**Final Value:** ${end_value:.2f}
**{profit_label}:** ${abs(profit):.2f}
**{return_label}:** {'+' if profit >= 0 else ''}{profit_pct:.1f}%

**Period:** {period}
**Shares Purchased:** {shares:.4f}
**Start Price:** ${start_price:.2f}
**End Price:** ${end_price:.2f}

PLOTLY_CHART:{fig_json}"""
        
        return result
        
    except Exception as e:
        return f"Error simulating investment for {ticker}: {str(e)}"
