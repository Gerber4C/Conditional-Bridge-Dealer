#!/usr/bin/env python3
"""
conditional_dealer.py
Bridge dealing simulator with extended suit-length features and North-hand precondition filtering.
Now supports efficient conditional generation: if a North shape condition is specified, the simulator directly constructs a North hand satisfying it, then deals the remaining cards randomly.

Usage examples:
  python conditional_dealer.py --deals 10000 --condition "Longest>=5 and HCP>=10"
  python conditional_dealer.py --deals 5000 --condition "SecondLongest==4 and HCP>=12"
  python conditional_dealer.py --north-shape "S==5 and H==3 and D==3 and C==2" --condition "HCP>=10"
  python conditional_dealer.py --deals 2000 --condition "Shortest<=2 and HCP>=8" --north-shape "Longest>=6"

Supported tokens in both conditions:
  HCP            : high-card points (A=4, K=3, Q=2, J=1)
  S, H, D, C     : suit lengths for Spades, Hearts, Diamonds, Clubs
  Longest        : length of the longest suit
  SecondLongest  : length of the 2nd longest suit
  ThirdLongest   : length of the 3rd longest suit
  Shortest       : length of the shortest suit
  Has XX         : whether holding card XX (e.g. Has_SA, Has_H10, Has_TD, Has_3C)

Supported operators:
  Comparison: == != > >= < <=
  Boolean: and, or, not
  Parentheses: ( )

Condition expressions are parsed and evaluated safely (no arbitrary code execution).
"""

import random
import argparse
import ast
from collections import Counter
import re

SUITS = ['S', 'H', 'D', 'C']
RANKS = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
HCP_MAP = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
DECK = [r + s for s in SUITS for r in RANKS]

def hand_suit_counts(hand):
    c = Counter(card[-1] for card in hand)
    return {s: c.get(s, 0) for s in SUITS}

def hand_hcp(hand):
    return sum(HCP_MAP.get(card[:-1], 0) for card in hand)

def suit_rankings(suits):
    lengths = sorted(suits.values(), reverse=True)
    return {
        'Longest': lengths[0],
        'SecondLongest': lengths[1],
        'ThirdLongest': lengths[2],
        'Shortest': lengths[3]
    }

ALLOWED_NODES = {
    'Expression', 'BoolOp', 'BinOp', 'UnaryOp', 'Compare', 'Name', 'Load',
    'Constant', 'And', 'Or', 'Not', 'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE',
    'Add', 'Sub', 'Mult', 'Div', 'Mod', 'USub', 'UAdd'
}

class ConditionEvaluator(ast.NodeVisitor):
    def __init__(self, context):
        self.context = context

    def visit(self, node):
        nodename = node.__class__.__name__
        if nodename not in ALLOWED_NODES:
            raise ValueError(f"Disallowed expression: {nodename}")
        return super().visit(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.And):
            return all(self.visit(v) for v in node.values)
        elif isinstance(node.op, ast.Or):
            return any(self.visit(v) for v in node.values)
        raise ValueError('Unsupported boolean operator')

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            return not bool(self.visit(node.operand))
        elif isinstance(node.op, ast.USub):
            return -self.visit(node.operand)
        elif isinstance(node.op, ast.UAdd):
            return +self.visit(node.operand)
        raise ValueError('Unsupported unary op')

    def visit_Compare(self, node):
        left = self.visit(node.left)
        for op_node, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if isinstance(op_node, ast.Eq):
                ok = left == right
            elif isinstance(op_node, ast.NotEq):
                ok = left != right
            elif isinstance(op_node, ast.Lt):
                ok = left < right
            elif isinstance(op_node, ast.LtE):
                ok = left <= right
            elif isinstance(op_node, ast.Gt):
                ok = left > right
            elif isinstance(op_node, ast.GtE):
                ok = left >= right
            else:
                raise ValueError('Unsupported comparison operator')
            if not ok:
                return False
            left = right
        return True

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        raise ValueError('Unsupported binary operator')

    def visit_Name(self, node):
        if node.id in self.context:
            return self.context[node.id]
        raise ValueError(f'Unknown name: {node.id}')

    def visit_Constant(self, node):
        return node.value

import re

CARD_ALIASES = {}
for s in SUITS:
    for r in RANKS:
        standard = r + s if r != '10' else 'T' + s
        # 支持两种顺序
        CARD_ALIASES[r + s] = standard
        CARD_ALIASES[s + r] = standard
        # 支持10的各种写法
        if r == '10':
            CARD_ALIASES['10' + s] = 'T' + s
            CARD_ALIASES[s + '10'] = 'T' + s
            CARD_ALIASES['T' + s] = 'T' + s
            CARD_ALIASES[s + 'T'] = 'T' + s

def card_to_standard(raw):
    raw = raw.upper().replace(" ", "")
    # 尝试各种顺序
    if raw in CARD_ALIASES:
        return CARD_ALIASES[raw]
    # 尝试拆解
    m = re.match(r'([SHDC])([AKQJT2-9]|10)$', raw)
    if m:
        return CARD_ALIASES.get(m.group(2) + m.group(1), raw)
    m = re.match(r'([AKQJT2-9]|10)([SHDC])$', raw)
    if m:
        return CARD_ALIASES.get(m.group(1) + m.group(2), raw)
    return raw

def normalize_condition(cond_str):
    # 替换 Has XX 或 HasXX 为 Has_标准牌名
    def has_replacer(match):
        raw = match.group(1) or match.group(2)
        std = card_to_standard(raw)
        return f"Has_{std}"

    # 支持空格和无空格
    cond_str = re.sub(r'Has\s*([AKQJ]|T|10|[2-9][SHDC])', has_replacer, cond_str, flags=re.IGNORECASE)
    cond_str = re.sub(r'Has\s*([SHDC][AKQJ]|[SHDC]T|[SHDC]10|[SHDC][2-9])', has_replacer, cond_str, flags=re.IGNORECASE)
    cond_str = re.sub(r'Has\s*([AKQJ]|T|10|[2-9])([SHDC])', lambda m: f"Has_{card_to_standard(m.group(1)+m.group(2))}", cond_str, flags=re.IGNORECASE)
    cond_str = re.sub(r'Has\s*([SHDC])([AKQJ]|T|10|[2-9])', lambda m: f"Has_{card_to_standard(m.group(2)+m.group(1))}", cond_str, flags=re.IGNORECASE)
    cond_str = cond_str.replace('AND', 'and').replace('OR', 'or').replace('NOT', 'not')
    return cond_str

def condition_matches(cond_str, hcp, suits, hand_cards=None):
    expr = normalize_condition(cond_str)
    suit_ranks = suit_rankings(suits)
    ctx = {
        'HCP': hcp,
        'S': suits['S'], 'H': suits['H'], 'D': suits['D'], 'C': suits['C'],
        'Longest': suit_ranks['Longest'],
        'SecondLongest': suit_ranks['SecondLongest'],
        'ThirdLongest': suit_ranks['ThirdLongest'],
        'Shortest': suit_ranks['Shortest']
    }
    if hand_cards is not None:
        # 统一10为T
        hand = set(card if not card.startswith('10') else 'T' + card[2:] for card in hand_cards)
        for std in set(CARD_ALIASES.values()):
            ctx[f'Has_{std}'] = std in hand
    tree = ast.parse(expr, mode='eval')
    evaluator = ConditionEvaluator(ctx)
    return bool(evaluator.visit(tree))

def parse_north_condition_for_weights(north_cond):
    # 简易解析: 返回目标花色、是否追求长套、HCP高低倾向
    cond = north_cond.replace(' ', '').upper()
    suit_bias = {s: 1.0 for s in SUITS}
    rank_bias = {r: 1.0 for r in RANKS}

    # 花色长套
    for s in SUITS:
        if f'{s}==' in cond or f'{s}>=' in cond or f'LONGEST>=' in cond or f'LONGEST==' in cond:
            suit_bias[s] += 2.0  # 增强指定花色权重

    # 检查 HCP 下界是否 >= 16
    # 用正则提取 HCP >= xx
    m = re.search(r'HCP\s*>=\s*(\d+)', cond)
    if m:
        hcp_min = int(m.group(1))
        if hcp_min >= 13:
            # 仅在 HCP >= 13 时赋值
            rank_bias['J'] = 669 * 4 / 2673
            rank_bias['Q'] = 822 * 4 / 2673
            rank_bias['K'] = 991 * 4 / 2673
            rank_bias['A'] = 1198 * 4 / 2673
        if hcp_min >= 16:
            # 仅在 HCP >= 16 时赋值
            rank_bias['J'] = 244 * 4 / 941
            rank_bias['Q'] = 339 * 4 / 941
            rank_bias['K'] = 403 * 4 / 941
            rank_bias['A'] = 540 * 4 / 941
        if hcp_min >= 19:
            # 仅在 HCP >= 19 时赋值
            rank_bias['J'] = 60 * 4 / 233
            rank_bias['Q'] = 96 * 4 / 233
            rank_bias['K'] = 134 * 4 / 233
            rank_bias['A'] = 176 * 4 / 233
        if hcp_min >= 22:
            # 仅在 HCP >= 22 时赋值
            rank_bias['J'] = 99 * 4 / 396
            rank_bias['Q'] = 185 * 4 / 396
            rank_bias['K'] = 201 * 4 / 396
            rank_bias['A'] = 279 * 4 / 396
    mi = re.search(r'HCP\s*<=\s*(\d+)', cond)
    if mi:
        hcp_max = int(m.group(1))
        if hcp_max <= 7:
            rank_bias['J'] = 662 * 4 / 2896
            rank_bias['Q'] = 504 * 4 / 2896
            rank_bias['K'] = 392 * 4 / 2896
            rank_bias['A'] = 222 * 4 / 2896
        if hcp_max <= 5:
            rank_bias['J'] = 300 * 4 / 1388
            rank_bias['Q'] = 213 * 4 / 1388
            rank_bias['K'] = 119 * 4 / 1388
            rank_bias['A'] = 58 * 4 / 1388
        if hcp_max <= 3:
            rank_bias['J'] = 102 * 4 / 522
            rank_bias['Q'] = 58 * 4 / 522
            rank_bias['K'] = 25 * 4 / 522
            rank_bias['A'] = 0
    return suit_bias, rank_bias

def weighted_sample(deck, weights, n):
    # 从deck中按weights抽取n张，不放回
    selected = []
    deck = deck[:]
    for _ in range(n):
        total_w = sum(weights[card] for card in deck)
        r = random.uniform(0, total_w)
        upto = 0
        for i, card in enumerate(deck):
            upto += weights[card]
            if upto >= r:
                selected.append(card)
                deck.pop(i)
                break
    return selected

def deal_one_with_north_condition(north_cond):
    deck = DECK[:]
    suit_bias, rank_bias = parse_north_condition_for_weights(north_cond)
    # 合成每张牌的抽取权重
    card_weights = {}
    for card in deck:
        s = card[-1]
        r = card[:-1]
        card_weights[card] = suit_bias[s] * rank_bias.get(r, 1.0)

    attempts = 0
    while True:
        attempts += 1
        # 用加权采样抽取北家
        north = weighted_sample(deck, card_weights, 13)
        suits_north = hand_suit_counts(north)
        hcp_north = hand_hcp(north)
        if condition_matches(north_cond, hcp_north, suits_north):
            break
        if attempts > 100000:
            raise RuntimeError(f"Could not find North hand satisfying condition after {attempts} tries.")
    remaining = [c for c in deck if c not in north]
    random.shuffle(remaining)
    east = remaining[:13]
    south = remaining[13:26]
    west  = remaining[26:39]
    return [north, east, south, west]

def simulate(deals, condition, north_shape=None, verbose=False):
    match_count = 0
    has_card_count = 0   # 新增：统计持有特定牌的次数
    # 提取所有 Has_XX
    has_cards = re.findall(r'Has\s+([AKQJ]0?|10|[2-9])[SHDC]', condition, flags=re.IGNORECASE)
    has_cards = [c.upper() for c in has_cards]
    for i in range(deals):
        if north_shape:
            hands = deal_one_with_north_condition(north_shape)
        else:
            deck = DECK[:]
            random.shuffle(deck)
            hands = [deck[i*13:(i+1)*13] for i in range(4)]

        south = hands[2]
        suits_south = hand_suit_counts(south)
        hcp_south = hand_hcp(south)

        if condition_matches(condition, hcp_south, suits_south, hand_cards=south):
            match_count += 1

        # 新增统计
        for card in has_cards:
            if card in south:
                has_card_count += 1

        if verbose and (i+1) % max(1, deals//10) == 0:
            print(f"Progress: {i+1}/{deals} deals, matches so far: {match_count}, {has_card_count} times Has {has_cards}")
    # 可在结果输出时增加统计
    return match_count, has_card_count

def main():
    # ...原参数处理不变...
    parser = argparse.ArgumentParser(description='Bridge deal simulator: count South hands matching a condition, optionally restricting North shape')
    parser.add_argument('--deals', '-n', type=int, default=10000, help='number of random deals to simulate')
    parser.add_argument('--condition', '-c', type=str, required=True, help='condition for South, e.g. "Longest>=5 and HCP>=10"')
    parser.add_argument('--north-shape', type=str, default=None, help='optional condition restricting North hand shape, e.g. "S==5 and H==3 and D==3 and C==2"')
    parser.add_argument('--seed', type=int, default=None, help='random seed (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='show progress')
    args = parser.parse_args()
    matches, has_card_count = simulate(args.deals, args.condition, north_shape=args.north_shape, verbose=args.verbose)
    pct = matches / args.deals * 100.0
    print(f"Matches: {matches} / {args.deals} ({pct:.4f}%)")
    # 新增输出
    has_cards = re.findall(r'Has\s+([AKQJ]0?|10|[2-9])[SHDC]', args.condition, flags=re.IGNORECASE)
    if has_cards:
        print(f"South held {has_cards} {has_card_count} times out of {args.deals} deals ({has_card_count / args.deals * 100:.2f}%)")

if __name__ == '__main__':
    main()
