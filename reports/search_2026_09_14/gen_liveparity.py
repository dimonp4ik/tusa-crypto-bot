"""Writes liveparity.py: does the LIVE signal code produce what the model measured?

Every number in this session comes from struct_params.bank_signals - a reimplementation of the bank's
hour scan written for speed. The bot itself calls pullback_bank.hourly_signals. Nobody has put the two
side by side. Twice before in this project the live path quietly failed to do what a passing check said
it did (a parity test passed on a knob that never fired in the real run; kNN sizing lived only in the
live bot), so a reimplementation agreeing with itself proves nothing.

Signal by signal, on the 14-coin live-like book, over the whole history and over the last 45 days:
which hours each side fires, which rule it picks, and the limit and ATR it hands the entry. Any hour
present on one side only, any different rule index, any price difference above a hundredth of a percent
is printed with its timestamp so it can be looked at directly.
"""
import ast
import io

s = io.open('widestopref.py', encoding='utf-8').read()
head = s[:s.index("for bi, (book_name, coins) in enumerate(BOOKS3):")]

TAIL = r'''
import datetime as DT
import tm_tag as T

BOOK = [c for c in SP.COINS if c not in ('BILLUSDT', 'AAVEUSDT')]
NOW = int(max(SP.CTX[BOOK[0]]['t15']))
D45 = NOW - 45 * 86400

print('  СВЕРКА ЖИВОГО КОДА С МОДЕЛЬЮ: %d монет, правила strict3+short2_wide, btc_sma=50' % len(BOOK), flush=True)
print('  модель: struct_params.bank_signals | живой код: pullback_bank.hourly_signals', flush=True)

tot_m = tot_l = tot_both = 0
only_m, only_l, diff_rule, diff_px = [], [], [], []
for s in BOOK:
    c = SP.CTX[s]
    mod = SP.bank_signals(RULES, s, 50)
    live = PB.hourly_signals(np.asarray(c['t15'], dtype=np.int64), c['a15'], T.bt, T.ba,
                             rules=RULES, regime=None, btc_sma=50)
    tot_m += len(mod); tot_l += len(live)
    for t in sorted(set(mod) | set(live)):
        a, b = mod.get(t), live.get(t)
        if a is None:
            only_l.append((t, s, b[0])); continue
        if b is None:
            only_m.append((t, s, a[0])); continue
        tot_both += 1
        if a[0] != b[0]:
            diff_rule.append((t, s, a[0], b[0]))
        d = max(abs(a[1] / b[1] - 1), abs(a[2] / b[2] - 1)) if b[1] and b[2] else 1.0
        if d > 1e-4:
            diff_px.append((t, s, a[1], b[1], a[2], b[2], d))

print('', flush=True)
print('  сигналов: модель %d, живой код %d, совпало по часу %d' % (tot_m, tot_l, tot_both), flush=True)
print('  есть только у модели: %d | есть только у живого кода: %d' % (len(only_m), len(only_l)), flush=True)
print('  разошлись в выборе правила: %d | разошлись в цене или ATR больше 0.01%%: %d'
      % (len(diff_rule), len(diff_px)), flush=True)


def fmt(t):
    return DT.datetime.fromtimestamp(t, DT.UTC).strftime('%Y-%m-%d %H:%M')


for lbl, arr in (('ТОЛЬКО У МОДЕЛИ', only_m), ('ТОЛЬКО У ЖИВОГО КОДА', only_l)):
    if arr:
        print('    %s (первые 10 из %d):' % (lbl, len(arr)), flush=True)
        for t, s, k in arr[:10]:
            print('      %s %-10s правило %d' % (fmt(t), s, k), flush=True)
        rec = [x for x in arr if x[0] >= D45]
        print('      из них за последние 45 дней: %d' % len(rec), flush=True)
for t, s, a, b in diff_rule[:10]:
    print('    РАЗНОЕ ПРАВИЛО %s %-10s модель %d, живой %d' % (fmt(t), s, a, b), flush=True)
for t, s, p1, p2, a1, a2, d in diff_px[:10]:
    print('    РАЗНАЯ ЦЕНА/ATR %s %-10s уровень %.6f против %.6f, ATR %.6f против %.6f (%.4f%%)'
          % (fmt(t), s, p1, p2, a1, a2, 100 * d), flush=True)

mr = sum(1 for t, s, k in only_m if t >= D45)
lr = sum(1 for t, s, k in only_l if t >= D45)
print('', flush=True)
print('  ПОСЛЕДНИЕ 45 ДНЕЙ (период листа сверки): расхождений только у модели %d, только у живого %d, '
      'разное правило %d, разная цена %d'
      % (mr, lr, sum(1 for x in diff_rule if x[0] >= D45), sum(1 for x in diff_px if x[0] >= D45)), flush=True)
print('  ВЕРДИКТ: %s' % ('живой код и модель дают ОДНО И ТО ЖЕ'
                         if not (only_m or only_l or diff_rule or diff_px)
                         else 'ЕСТЬ РАСХОЖДЕНИЯ, смотреть выше'), flush=True)
'''

src = head + TAIL
ast.parse(src)
io.open('liveparity.py', 'w', encoding='utf-8').write(src)
print('liveparity.py gotov, sintaksis ok')
