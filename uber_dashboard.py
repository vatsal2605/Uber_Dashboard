import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle, Arc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

# ── Load & prep ──────────────────────────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/uploads/Uber_Dataset.csv')
df['START_DATE'] = pd.to_datetime(df['START_DATE'], format='%d-%m-%Y %H:%M', errors='coerce')
df['END_DATE']   = pd.to_datetime(df['END_DATE'],   format='%d-%m-%Y %H:%M', errors='coerce')
df['DURATION_MIN'] = (df['END_DATE'] - df['START_DATE']).dt.total_seconds() / 60
df = df[df['CATEGORY'].isin(['Business', 'Personal'])].copy()
df['MONTH']    = df['START_DATE'].dt.to_period('M')
df['HOUR']     = df['START_DATE'].dt.hour
df['WEEKDAY']  = df['START_DATE'].dt.day_name()
df['DIST']     = df['DISTANCE(in kms)']

# ── Palette ──────────────────────────────────────────────────────────────────
BG        = '#0A0E1A'
CARD      = '#111827'
CARD2     = '#141E2E'
BORDER    = '#1E293B'
CYAN      = '#00E5FF'
MAGENTA   = '#FF2D78'
AMBER     = '#FFB800'
LIME      = '#39FF14'
VIOLET    = '#A855F7'
SLATE     = '#94A3B8'
WHITE     = '#F1F5F9'
GRID      = '#1E293B'

ACCENT_PALETTE = [CYAN, MAGENTA, AMBER, LIME, VIOLET, '#FF6B35', '#00FFB3']

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 22), facecolor=BG)
fig.patch.set_facecolor(BG)

outer = gridspec.GridSpec(
    5, 4,
    figure=fig,
    hspace=0.55, wspace=0.38,
    left=0.04, right=0.97,
    top=0.91, bottom=0.04,
    height_ratios=[1, 2.2, 2.2, 2.2, 1.1]
)

# ── Helper: card background ───────────────────────────────────────────────────
def card_bg(ax, color=CARD, alpha=1.0):
    ax.set_facecolor(color)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
        sp.set_linewidth(1.2)

def label(ax, txt, fs=9, color=SLATE, loc='left', pad=6):
    ax.set_title(txt, fontsize=fs, color=color, loc=loc, pad=pad,
                 fontweight='bold', fontfamily='monospace')

def glow(text_obj, color, lw=4):
    text_obj.set_path_effects([
        pe.withStroke(linewidth=lw, foreground=color + '55'),
        pe.Normal()
    ])

# ═══════════════════════════════════════════════════════════════════════════
# TOP HEADER STRIP
# ═══════════════════════════════════════════════════════════════════════════
header_ax = fig.add_axes([0.0, 0.93, 1.0, 0.07], facecolor=BG)
header_ax.set_xlim(0, 1); header_ax.set_ylim(0, 1)
header_ax.axis('off')

# Gradient accent bar
for i, x in enumerate(np.linspace(0, 1, 500)):
    c = plt.cm.cool(i / 500)
    header_ax.axvline(x, color=c, alpha=0.6, linewidth=0.8)

t = header_ax.text(0.5, 0.55, '⬡  UBER RIDES  —  ANALYTICS DASHBOARD',
    ha='center', va='center', fontsize=22, fontweight='bold',
    color=WHITE, fontfamily='monospace', transform=header_ax.transAxes)
glow(t, CYAN, 6)

header_ax.text(0.5, 0.08, 'Business & Personal Trip Intelligence  |  2026 Dataset  |  India Operations',
    ha='center', va='bottom', fontsize=9, color=SLATE,
    fontfamily='monospace', transform=header_ax.transAxes)

# ═══════════════════════════════════════════════════════════════════════════
# KPI CARDS — row 0
# ═══════════════════════════════════════════════════════════════════════════
kpis = [
    ('TOTAL TRIPS',     f"{len(df):,}",            CYAN,    '🚗'),
    ('TOTAL DISTANCE',  f"{df['DIST'].sum():,.0f} km",  MAGENTA, '📍'),
    ('AVG DISTANCE',    f"{df['DIST'].mean():.1f} km",   AMBER,   '📏'),
    ('AVG DURATION',    f"{df['DURATION_MIN'].mean():.0f} min", LIME, '⏱'),
]

for col, (title, val, accent, icon) in enumerate(kpis):
    ax = fig.add_subplot(outer[0, col])
    card_bg(ax, CARD)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # accent strip on left
    ax.add_patch(FancyBboxPatch((0, 0), 0.04, 1,
        boxstyle='square,pad=0', facecolor=accent, alpha=0.85, zorder=3))

    ax.text(0.12, 0.72, icon, fontsize=22, va='center', transform=ax.transAxes)
    ax.text(0.12, 0.42, val, fontsize=20, fontweight='bold',
            color=accent, fontfamily='monospace', transform=ax.transAxes)
    glow(ax.texts[-1], accent, 3)
    ax.text(0.12, 0.20, title, fontsize=7.5, color=SLATE,
            fontfamily='monospace', transform=ax.transAxes)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 1 — Monthly trend (span 3) + Donut (span 1)
# ═══════════════════════════════════════════════════════════════════════════

# ── Monthly trip trend ────────────────────────────────────────────────────
ax_trend = fig.add_subplot(outer[1, :3])
card_bg(ax_trend, CARD2)
label(ax_trend, '  MONTHLY TRIP VOLUME & DISTANCE TREND', 9.5)

monthly = df.groupby('MONTH').agg(
    trips=('DIST', 'count'),
    dist=('DIST', 'sum')
).reset_index()
monthly['MONTH_STR'] = monthly['MONTH'].astype(str)

x = np.arange(len(monthly))
ax_trend.bar(x, monthly['trips'], color=CYAN, alpha=0.25, width=0.6)
ax_trend.bar(x, monthly['trips'], color=CYAN, alpha=0.0, width=0.6)  # placeholder

# Filled area
ax_trend.fill_between(x, monthly['trips'], alpha=0.18, color=CYAN)
line1, = ax_trend.plot(x, monthly['trips'], color=CYAN, lw=2.5, marker='o',
                        markersize=5, markerfacecolor=BG, markeredgecolor=CYAN,
                        markeredgewidth=2, zorder=5, label='Trips')

ax2 = ax_trend.twinx()
ax2.fill_between(x, monthly['dist'], alpha=0.12, color=MAGENTA)
line2, = ax2.plot(x, monthly['dist'], color=MAGENTA, lw=2, linestyle='--',
                   marker='s', markersize=4, markerfacecolor=BG,
                   markeredgecolor=MAGENTA, markeredgewidth=2, label='Distance (km)')

ax_trend.set_facecolor(CARD2)
ax2.set_facecolor(CARD2)
ax_trend.set_xticks(x)
ax_trend.set_xticklabels(monthly['MONTH_STR'], rotation=35, ha='right',
                          fontsize=7.5, color=SLATE, fontfamily='monospace')
ax_trend.tick_params(colors=SLATE, labelsize=8)
ax2.tick_params(colors=MAGENTA, labelsize=8)
ax_trend.yaxis.label.set_color(CYAN)
ax_trend.set_ylabel('Trips', color=CYAN, fontsize=8, fontfamily='monospace')
ax2.set_ylabel('Distance (km)', color=MAGENTA, fontsize=8, fontfamily='monospace')
ax_trend.grid(axis='y', color=GRID, linewidth=0.6, alpha=0.8)
ax_trend.set_axisbelow(True)
for sp in ax_trend.spines.values(): sp.set_color(BORDER)
for sp in ax2.spines.values(): sp.set_color(BORDER)

lines = [line1, line2]
ax_trend.legend(lines, [l.get_label() for l in lines],
                 loc='upper left', facecolor=CARD, edgecolor=BORDER,
                 labelcolor=WHITE, fontsize=8, framealpha=0.8)

# ── Donut: Business vs Personal ───────────────────────────────────────────
ax_donut = fig.add_subplot(outer[1, 3])
card_bg(ax_donut, CARD2)
label(ax_donut, '  TRIP CATEGORY', 9.5)

cat_counts = df['CATEGORY'].value_counts()
donut_colors = [CYAN, MAGENTA]
wedges, _ = ax_donut.pie(
    cat_counts.values,
    colors=donut_colors,
    startangle=90,
    wedgeprops=dict(width=0.55, edgecolor=CARD2, linewidth=3),
    radius=0.85
)
for w, c in zip(wedges, donut_colors):
    w.set_gid(c)

ax_donut.text(0, 0.08, f'{len(df)}', ha='center', va='center',
               fontsize=17, fontweight='bold', color=WHITE,
               fontfamily='monospace')
ax_donut.text(0, -0.18, 'TRIPS', ha='center', va='center',
               fontsize=7, color=SLATE, fontfamily='monospace')

# Legend below the donut — fully inside axes bounds
for i, (cat, cnt, col) in enumerate(zip(cat_counts.index, cat_counts.values, donut_colors)):
    pct = cnt / cat_counts.sum() * 100
    y_pos = -1.25 - i * 0.32
    ax_donut.text(0, y_pos, f'● {cat}', fontsize=8.5, color=col,
                   fontfamily='monospace', ha='center', va='center', fontweight='bold')
    ax_donut.text(0, y_pos - 0.18, f'{cnt:,}  ({pct:.1f}%)', fontsize=8,
                   color=WHITE, fontfamily='monospace', ha='center', va='center')

ax_donut.set_ylim(-2.1, 1.1)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 2 — Purpose bar (2) + Hourly heatmap (2)
# ═══════════════════════════════════════════════════════════════════════════

# ── Purpose bar chart ─────────────────────────────────────────────────────
ax_purpose = fig.add_subplot(outer[2, :2])
card_bg(ax_purpose, CARD)
label(ax_purpose, '  TRIPS BY PURPOSE', 9.5)

purpose = df['PURPOSE'].value_counts().dropna().head(9)
colors_p = ACCENT_PALETTE[:len(purpose)]
bars = ax_purpose.barh(purpose.index[::-1], purpose.values[::-1],
                        color=colors_p[::-1], height=0.62,
                        edgecolor=BG, linewidth=0.8)

for bar, val, col in zip(bars, purpose.values[::-1], colors_p[::-1]):
    bar.set_alpha(0.85)
    ax_purpose.text(val + 3, bar.get_y() + bar.get_height() / 2,
                     f'{val}', va='center', ha='left',
                     color=col, fontsize=8, fontweight='bold',
                     fontfamily='monospace')

ax_purpose.set_facecolor(CARD)
ax_purpose.tick_params(colors=SLATE, labelsize=8)
ax_purpose.set_xlabel('Number of Trips', color=SLATE, fontsize=8,
                        fontfamily='monospace')
ax_purpose.grid(axis='x', color=GRID, linewidth=0.5, alpha=0.7)
ax_purpose.set_axisbelow(True)
for sp in ax_purpose.spines.values(): sp.set_color(BORDER)
ax_purpose.tick_params(axis='y', colors=WHITE, labelsize=8)

# ── Hourly trip distribution ──────────────────────────────────────────────
ax_hour = fig.add_subplot(outer[2, 2:])
card_bg(ax_hour, CARD)
label(ax_hour, '  HOURLY TRIP DISTRIBUTION BY WEEKDAY', 9.5)

weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
heat_data = df.groupby(['WEEKDAY','HOUR']).size().unstack(fill_value=0)
# Ensure all 24 hours present
heat_data = heat_data.reindex(columns=range(24), fill_value=0)
heat_data = heat_data.reindex(weekday_order, fill_value=0)

cmap = LinearSegmentedColormap.from_list('neon', [BG, VIOLET, CYAN, AMBER], N=256)
im = ax_hour.imshow(heat_data.values, aspect='auto', cmap=cmap,
                     interpolation='nearest', vmin=0)

# ── Time-block shading bands (subtle vertical stripes) ────────────────────
time_blocks = [
    (0,  5,  '#FFFFFF08', 'Early\nMorning\n12AM–5AM'),
    (6,  11, '#00E5FF0A', 'Morning\n6AM–11AM'),
    (12, 17, '#FFB8000A', 'Afternoon\n12PM–5PM'),
    (18, 21, '#FF2D780D', 'Evening\n6PM–9PM'),
    (22, 23, '#A855F70A', 'Night\n10PM–11PM'),
]
for start, end, shade, blk_label in time_blocks:
    ax_hour.axvspan(start - 0.5, end + 0.5, color=shade, zorder=0)
    # block label above the heatmap
    mid = (start + end) / 2
    ax_hour.text(mid, -0.85, blk_label, ha='center', va='top',
                  fontsize=6, color=SLATE, fontfamily='monospace',
                  transform=ax_hour.get_xaxis_transform(), linespacing=1.3)

# ── X-axis: every hour labelled with AM/PM notation ──────────────────────
hour_labels = []
for h in range(24):
    if h == 0:
        hour_labels.append('12AM')
    elif h < 12:
        hour_labels.append(f'{h}AM')
    elif h == 12:
        hour_labels.append('12PM')
    else:
        hour_labels.append(f'{h-12}PM')

ax_hour.set_xticks(range(24))
ax_hour.set_xticklabels(hour_labels, fontsize=6.5, color=WHITE,
                          fontfamily='monospace', rotation=45, ha='right')

# ── Y-axis: full weekday names ────────────────────────────────────────────
ax_hour.set_yticks(range(len(weekday_order)))
ax_hour.set_yticklabels(weekday_order, fontsize=8.5, color=WHITE,
                          fontfamily='monospace', fontweight='bold')

# ── Divider lines between days ────────────────────────────────────────────
for y in np.arange(0.5, 6.5, 1):
    ax_hour.axhline(y, color=BORDER, linewidth=0.8, alpha=0.6)

# ── Divider lines between time blocks ─────────────────────────────────────
for x_div in [5.5, 11.5, 17.5, 21.5]:
    ax_hour.axvline(x_div, color=BORDER, linewidth=1, alpha=0.9, linestyle='--')

# ── AM / PM marker labels on top ──────────────────────────────────────────
ax_hour.text(5.75,  -0.06, '◀ AM', ha='left',  va='center', fontsize=7,
              color=CYAN+'99', fontfamily='monospace',
              transform=ax_hour.get_xaxis_transform())
ax_hour.text(11.75, -0.06, 'PM ▶', ha='left', va='center', fontsize=7,
              color=AMBER+'99', fontfamily='monospace',
              transform=ax_hour.get_xaxis_transform())

cbar = plt.colorbar(im, ax=ax_hour, fraction=0.025, pad=0.02)
cbar.ax.tick_params(colors=SLATE, labelsize=7)
cbar.ax.set_ylabel('No. of Trips', color=SLATE, fontsize=7, fontfamily='monospace')
ax_hour.tick_params(which='both', length=0)
ax_hour.set_xlabel('Hour of Day  (AM / PM)', color=SLATE, fontsize=8,
                    fontfamily='monospace', labelpad=28)
for sp in ax_hour.spines.values(): sp.set_color(BORDER)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 3 — Top Cities (2) + Distance Histogram (2)
# ═══════════════════════════════════════════════════════════════════════════

# ── Top cities bar ────────────────────────────────────────────────────────
ax_city = fig.add_subplot(outer[3, :2])
card_bg(ax_city, CARD2)
label(ax_city, '  TOP START CITIES', 9.5)

cities = df['START'].value_counts().head(6)
x_c = np.arange(len(cities))
city_colors = [CYAN, MAGENTA, AMBER, LIME, VIOLET, '#FF6B35']

bars_c = ax_city.bar(x_c, cities.values, color=city_colors,
                      width=0.6, edgecolor=BG, linewidth=1)

for bar, val, col in zip(bars_c, cities.values, city_colors):
    bar.set_alpha(0.88)
    ax_city.text(bar.get_x() + bar.get_width() / 2, val + 2,
                  f'{val}', ha='center', va='bottom',
                  color=col, fontsize=9, fontweight='bold',
                  fontfamily='monospace')

# bottom accent line
ax_city.axhline(0, color=BORDER, linewidth=1.5)

ax_city.set_facecolor(CARD2)
ax_city.set_xticks(x_c)
ax_city.set_xticklabels(cities.index, fontsize=8.5, color=WHITE,
                          fontfamily='monospace')
ax_city.tick_params(colors=SLATE, labelsize=8)
ax_city.set_ylabel('Trips', color=SLATE, fontsize=8, fontfamily='monospace')
ax_city.grid(axis='y', color=GRID, linewidth=0.5, alpha=0.7)
ax_city.set_axisbelow(True)
for sp in ax_city.spines.values(): sp.set_color(BORDER)

# ── Distance distribution ─────────────────────────────────────────────────
ax_dist = fig.add_subplot(outer[3, 2:])
card_bg(ax_dist, CARD2)
label(ax_dist, '  TRIP DISTANCE DISTRIBUTION (≤ 60 km)', 9.5)

dist_filtered = df[df['DIST'] <= 60]['DIST'].dropna()
n, bins, patches = ax_dist.hist(dist_filtered, bins=35, edgecolor=BG,
                                  linewidth=0.5)

# Color gradient on bars
cmap_hist = LinearSegmentedColormap.from_list('hist', [MAGENTA, AMBER, LIME], N=len(patches))
for i, (patch, pct) in enumerate(zip(patches, np.linspace(0, 1, len(patches)))):
    patch.set_facecolor(cmap_hist(pct))
    patch.set_alpha(0.88)

# KDE-style line
from scipy.stats import gaussian_kde
kde = gaussian_kde(dist_filtered, bw_method=0.3)
x_kde = np.linspace(dist_filtered.min(), dist_filtered.max(), 300)
y_kde = kde(x_kde) * len(dist_filtered) * (bins[1] - bins[0])
ax_dist.plot(x_kde, y_kde, color=CYAN, lw=2.2, zorder=5)
ax_dist.fill_between(x_kde, y_kde, alpha=0.08, color=CYAN)

ax_dist.set_facecolor(CARD2)
ax_dist.tick_params(colors=SLATE, labelsize=8)
ax_dist.set_xlabel('Distance (km)', color=SLATE, fontsize=8, fontfamily='monospace')
ax_dist.set_ylabel('Frequency', color=SLATE, fontsize=8, fontfamily='monospace')
ax_dist.grid(axis='y', color=GRID, linewidth=0.5, alpha=0.7)
ax_dist.set_axisbelow(True)
for sp in ax_dist.spines.values(): sp.set_color(BORDER)

# median line
med = dist_filtered.median()
ax_dist.axvline(med, color=AMBER, linestyle='--', linewidth=1.5, alpha=0.9)
ax_dist.text(med + 0.5, ax_dist.get_ylim()[1] * 0.88,
              f'Median\n{med:.1f} km', color=AMBER, fontsize=7.5,
              fontfamily='monospace', va='top')

# ═══════════════════════════════════════════════════════════════════════════
# ROW 4 — INSIGHTS PANEL (full width)
# ═══════════════════════════════════════════════════════════════════════════
ax_ins = fig.add_subplot(outer[4, :])
ax_ins.set_facecolor(CARD2)
for sp in ax_ins.spines.values():
    sp.set_color(CYAN + '55')
    sp.set_linewidth(1.2)
ax_ins.set_xlim(0, 1); ax_ins.set_ylim(0, 1); ax_ins.axis('off')

# Section title
ax_ins.text(0.0, 0.92, '  💡  KEY INSIGHTS  &  BUSINESS RECOMMENDATIONS',
             fontsize=10, fontweight='bold', color=CYAN,
             fontfamily='monospace', va='top', transform=ax_ins.transAxes)
ax_ins.plot([0, 1], [0.78, 0.78], color=CYAN + '33', linewidth=0.8,
             transform=ax_ins.transAxes, clip_on=False)

# ── Compute real insights from data ──────────────────────────────────────
# Peak hour
hourly_trips = df.groupby('HOUR').size()
peak_hour    = int(hourly_trips.idxmax())
peak_h_label = f"{peak_hour % 12 or 12}{'AM' if peak_hour < 12 else 'PM'}"
peak_end     = f"{(peak_hour+1) % 12 or 12}{'AM' if (peak_hour+1) < 12 else 'PM'}"

# Evening surge (6PM-9PM)
eve_trips  = df[df['HOUR'].between(18, 21)].shape[0]
eve_pct    = eve_trips / len(df) * 100

# Busiest weekday
day_trips  = df.groupby('WEEKDAY').size()
busiest_day = day_trips.idxmax()

# Top purpose
top_purpose = df['PURPOSE'].value_counts().idxmax()
top_pur_pct = df['PURPOSE'].value_counts().iloc[0] / len(df) * 100

# Business dominance
biz_pct = df[df['CATEGORY'] == 'Business'].shape[0] / len(df) * 100

# Shortest avg trip city
city_avg = df.groupby('START')['DIST'].mean()
shortest_city = city_avg.idxmin()
shortest_km   = city_avg.min()

insights = [
    (CYAN,    '🕕',
     f'Peak Hour: {peak_h_label} – {peak_end}',
     f'{eve_pct:.0f}% of all trips fall in 6 PM–9 PM.\nConsider dynamic pricing\nduring evening surge.'),
    (AMBER,   '📅',
     f'Busiest Day: {busiest_day}',
     f'{busiest_day}s record the\nhighest trip volume.\nSchedule driver incentives\non this day.'),
    (MAGENTA, '🎯',
     f'Top Purpose: {top_purpose}',
     f'{top_pur_pct:.0f}% of trips are for\n"{top_purpose}".\nTarget promotions around\nthis use case.'),
    (LIME,    '💼',
     f'Business Trips: {biz_pct:.0f}%',
     f'Nearly all rides are\nbusiness-related.\nCorporate packages &\nmonthly plans recommended.'),
    (VIOLET,  '📍',
     f'Efficient City: {shortest_city}',
     f'Avg trip only {shortest_km:.1f} km.\nHigh ride frequency\npossible — ideal for\nfleet optimisation.'),
]

n_ins    = len(insights)
col_w    = 1.0 / n_ins
pad      = 0.012

for i, (color, icon, heading, body) in enumerate(insights):
    x0 = i * col_w + pad
    # card background
    ax_ins.add_patch(FancyBboxPatch(
        (x0, 0.06), col_w - pad * 2, 0.66,
        boxstyle='round,pad=0.01',
        facecolor=color + '12', edgecolor=color + '44',
        linewidth=1, transform=ax_ins.transAxes, zorder=2
    ))
    # left accent bar
    ax_ins.add_patch(FancyBboxPatch(
        (x0, 0.06), 0.003, 0.66,
        boxstyle='square,pad=0',
        facecolor=color, alpha=0.9,
        transform=ax_ins.transAxes, zorder=3
    ))
    cx = x0 + 0.008
    # icon
    ax_ins.text(cx, 0.65, icon, fontsize=13, va='center',
                 transform=ax_ins.transAxes, zorder=4)
    # heading
    t_h = ax_ins.text(cx, 0.52, heading, fontsize=7.8, fontweight='bold',
                       color=color, fontfamily='monospace', va='top',
                       transform=ax_ins.transAxes, zorder=4)
    # body
    ax_ins.text(cx, 0.38, body, fontsize=6.8, color=WHITE,
                 fontfamily='monospace', va='top', linespacing=1.5,
                 transform=ax_ins.transAxes, zorder=4)

# ── Footer watermark ──────────────────────────────────────────────────────
fig.text(0.97, 0.01, 'Data Analytics Portfolio  |  Built with Python & Matplotlib',
          ha='right', va='bottom', fontsize=7.5, color=SLATE,
          fontfamily='monospace', alpha=0.7)

fig.text(0.03, 0.01, '◈  Uber Rides Dataset  |  India  |  2026',
          ha='left', va='bottom', fontsize=7.5, color=SLATE,
          fontfamily='monospace', alpha=0.7)

# ── Save ──────────────────────────────────────────────────────────────────
plt.savefig('/mnt/user-data/outputs/uber_dashboard.png',
            dpi=180, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
print("Saved.")
