from time import perf_counter
from tqdm import tqdm
import numpy as np
import types

def map_id_to_card(id:int): return (id // 13, id % 13) # returns (suit, rank)

class Hand():
    def __init__(
            self, 
            given = None,
            **_
        ):

        if isinstance(given, Hand): # copies input hand
            self.array_rep = given.array_rep.copy()
            self.fixed = given.fixed.copy()
        elif given is not None:
            assert given.shape == (4,4,13), "Given hand must be a (4,4,13) numpy array."
            self.array_rep = given.astype('?')
            self.fixed = given.any(axis = 0) # axes 0,1 = suit, card
        else: 
            self.array_rep = np.zeros((4,4,13)).astype('?') # axes 0,1,2 = player, suit, card
            self.fixed = np.zeros((4,13)).astype('?') # axes 0,1 = suit, card
        self.shape = self._get_shape() # axes 0,1 = player, suit
        self.hcp = self._get_hcp() # axes 0 = player
        self.pbn = ''

    def _dealt(self):
        return self.array_rep.sum() == 52

    def _get_shape(self):
        return self.array_rep.sum(axis=2)
    
    def _get_hcp(self):
        hcp_filter = np.zeros((4,13)).astype(int)
        hcp_filter[:,:4] = np.repeat([[4,3,2,1]], 4, axis = 0)
        self.hcp = (self.array_rep * hcp_filter).sum(axis=(1,2))
        return self.hcp

    def _get_pbn(self):
        card_filter = np.array(['A','K','Q','J','T','9','8','7','6','5','4','3','2'])
        pbn = []
        for i in range(4):
            tmp = []
            for j in range(4): tmp.append(''.join(card_filter[self.array_rep[i,j]]))
            pbn.append('.'.join(tmp))
        self.pbn = '"N:'+' '.join(pbn)+'"'
        return self.pbn
    
    def reset(self):
        self.array_rep[:, ~self.fixed] = False

    def copy(self):
        return Hand(given = self.array_rep.copy())

class SuitPermuter():
    def __init__(self, overall = False, **kwargs):
        self.s = overall; self.h = overall; self.d = overall; self.c = overall
        for k,v in kwargs.items():
            if k not in ['s','h','d','c']: continue
            if not isinstance(v, bool): raise ValueError(f"SuitPermuter argument {k} must be a boolean.")
            setattr(self, k, v)
        
        permutable = []
        if self.s: permutable.append(0)
        if self.h: permutable.append(1)
        if self.d: permutable.append(2)
        if self.c: permutable.append(3)
        self.permutable = permutable
    
    def permute(self, hand: Hand):
        perm = np.arange(4)
        np.random.shuffle(perm[self.permutable])
        hand.array_rep = hand.array_rep[:, perm, :]
        hand.shape = hand.shape[:, perm]
        return hand

class HCPConstraint():
    def __init__(
        self,
        hcp_max:np.ndarray = np.ones(4) * 37,
        hcp_min:np.ndarray = np.zeros(4),
        **_ # additional params are ignored
    ):
        
        hcp_max = np.array(hcp_max); hcp_min = np.array(hcp_min)
        # sense check
        for cond in [hcp_max, hcp_min]:
            assert len(cond) == 4, "Each condition array must have exactly 4 elements."
        assert (37 >= hcp_max).all() and (hcp_max >= hcp_min).all() and (hcp_min >= 0).all(), \
            '0 <= min HCP <= max HCP <= 37'
        assert sum(hcp_max) >= 40 and sum(hcp_min) <= 40, 'Total HCP must be exactly 40'
        self.hcp_max = hcp_max.astype(int) # if hcp_max != np.ones(4) * 37 else None
        self.hcp_min = hcp_min.astype(int) # if hcp_min != np.zeros(4) else None

        # flag for quicker dealing
        self.no_hcp_cond = (hcp_max >= 37).all() and (hcp_min == 0).all()
    
    def check(self, hand: Hand):
        if self.no_hcp_cond: return True
        _ = hand._get_hcp() # forcibly updates the HCP attribute of the hand
        return all(hand.hcp <= self.hcp_max) and all(hand.hcp >= self.hcp_min)

class ShapeConstraint():
    def __new__(
        self,
        shape_max = np.ones((4,4))*13, # axis 0 = hand, axis 1 = suit
        shape_min = np.zeros((4,4)),
        permute_suits: SuitPermuter | bool = False, # True if you want to set longest suit etc.
        **_ # additional params are ignored
    ):
        shape_max = np.array(shape_max); shape_min = np.array(shape_min)
        # sense check
        for cond in [shape_max, shape_min]:
            assert cond.shape==(4,4), 'Shape specifications should be (4,4) arrays'
        sum_max = shape_max.sum(axis=0)
        sum_min = shape_min.sum(axis=0)
        assert np.all(sum_max >= 13), "The sum of max cards in all suits must be at least 13."
        assert np.all(sum_min <= 13), "The sum of min cards in all suits must be less than or equal to 13."

        # check if conditions are only on a single hand
        no_cond = np.zeros(4).astype('?')
        for i in range(4):
            if (shape_max[i,:] == 13).all() and (shape_min[i,:] == 0).all(): no_cond[i] = True
        if sum(no_cond) == 3:
            idx = np.where(no_cond == False)[0][0]
            return SingleHandShapeConstraint(
                idx, shape_max[idx,:], shape_min[idx,:], permute_suits = permute_suits
            )

        # set attributes
        # for cond, attr in zip([s_max, h_max, d_max, c_max], ['s_max', 'h_max', 'd_max', 'c_max']):
        #     cond = np.array(cond)
        #     setattr(self, attr, cond if cond != np.repeat(13, 4) else None)
        
        # for cond, attr in zip([s_min, h_min, d_min, c_min], ['s_min', 'h_min', 'd_min', 'c_min']):
        #     cond = np.array(cond)
        #     setattr(self, attr, cond if cond != np.zeros(4) else None)
        
        self.shape_max = shape_max.astype(int)
        self.shape_min = shape_min.astype(int)
        self.permute_suits = SuitPermuter(permute_suits) if isinstance(permute_suits, bool) else permute_suits

        # flag for quicker dealing
        self.no_shape_cond = (self.shape_max == 13).all() and (self.shape_min == 0).all()
        return self
    
    def check(self, hand: Hand):
        if self.no_shape_cond: return True
        _ = hand._get_shape() # forcibly updates the shape attribute of the hand
        return (hand.shape <= self.shape_max).all() and (hand.shape >= self.shape_min).all()

class SingleHandShapeConstraint():
    def __init__(
        self,
        hand : int = 0, # 0 = North, 1 = East, 2 = South, 3 = West
        suit_max:np.ndarray = np.repeat(13, 4),
        suit_min:np.ndarray = np.zeros(4),
        permute_suits = False, # True if you want to set longest suit etc.
        **_ # additional params are ignored
    ):
        if isinstance(suit_max, (int, float)): suit_max = np.repeat(suit_max, 4).astype(int)
        if isinstance(suit_min, (int, float)): suit_min = np.repeat(suit_min, 4).astype(int)
        suit_max = np.array(suit_max); suit_min = np.array(suit_min)
        # sense check
        assert 0 <= hand <= 3, 'Hand must be an integer between 0 and 3.'
        assert (13 >= suit_max).all() and (suit_max >= suit_min).all() and (suit_min >= 0).all(), \
            '0 <= min cards <= max cards <= 13'
        assert sum(suit_max) >= 13 and sum(suit_min) <= 13, 'Hands must be exactly 13 cards'
        
        self.hand = hand
        self.suit_min = suit_min.astype(int); self.suit_max = suit_max.astype(int)
        self.permute_suits = permute_suits if isinstance(permute_suits, SuitPermuter) else SuitPermuter(permute_suits)

        # flag for quicker dealing
        self.no_shape_cond = (suit_max == 13).all() and (suit_min == 0).all()

    def check(self, hand: Hand):
        if self.no_shape_cond: return True
        _ = hand._get_shape() # forcibly updates the shape attribute of the hand
        return (hand.shape[self.hand,:] <= self.suit_max).all() and (hand.shape[self.hand,:] >= self.suit_min).all()
    
class Constraint():
    def __init__(
        self,
        shape_constraint : ShapeConstraint | SingleHandShapeConstraint | types.NoneType = None,
        hcp_constraint : HCPConstraint | types.NoneType = None,
        **kwargs
    ):
        if shape_constraint is not None: self.shape_constraint = shape_constraint
        elif len(kwargs) > 0: self.shape_constraint = ShapeConstraint(**kwargs)
        else: self.shape_constraint = None
        if hcp_constraint is not None: self.hcp_constraint = hcp_constraint
        elif len(kwargs) > 0: self.hcp_constraint = HCPConstraint(**kwargs)
        else: self.hcp_constraint = None
        self.no_shape_cond = self.shape_constraint is None or self.shape_constraint.no_shape_cond
        self.no_hcp_cond = self.hcp_constraint is None or self.hcp_constraint.no_hcp_cond
        self.permute_suits = self.shape_constraint.permute_suits if self.shape_constraint is not None else SuitPermuter()

    def _parse_string(self):
        # TODO
        pass

    def check(self, hand: Hand):
        if not self.no_shape_cond and not self.shape_constraint.check(hand): return False
        if not self.no_hcp_cond and not self.hcp_constraint.check(hand): return False
        return True
    
class Dealer():
    def __init__(
            self, 
            constraint: Constraint | ShapeConstraint | SingleHandShapeConstraint | HCPConstraint | types.NoneType = None,
            given: np.ndarray | Hand | types.NoneType = None,
            **kwargs
        ):
        if isinstance(constraint, Constraint): self.constraint = constraint
        elif isinstance(constraint, (ShapeConstraint, SingleHandShapeConstraint)): 
            self.constraint = Constraint(shape_constraint = constraint, **kwargs)
        elif isinstance(constraint, HCPConstraint): 
            self.constraint = Constraint(hcp_constraint = constraint, **kwargs)
        else: self.constraint = Constraint(**kwargs)

        if given is not None: self.partial_deal = Hand(given = given)
        else: self.partial_deal = Hand()

    # deals all remaining cards from a partial deal
    def _random(self, hand:Hand | types.NoneType = None):
        if hand is None: hand = self.partial_deal.copy()
        available_cards = np.stack(np.where(~hand.array_rep.any(axis=0))).T # (suit, rank)
        np.random.shuffle(available_cards)
        remaining_cards = np.ones(4)*13 - hand.array_rep.sum(axis=(1,2))
        for idx, card in enumerate(available_cards):
            if idx < remaining_cards[0]: hand.array_rep[0, card[0], card[1]] = True
            elif idx < remaining_cards[0] + remaining_cards[1]: hand.array_rep[1, card[0], card[1]] = True
            elif idx < remaining_cards[0:3].sum(): hand.array_rep[2, card[0], card[1]] = True
            else: hand.array_rep[3, card[0], card[1]] = True
        return hand

    # deals all hands according to a single-hand constraint
    def _deal_shape_single_hand_constraint(
            self, 
            hand: Hand | types.NoneType = None,
            constraint: Constraint | SingleHandShapeConstraint | types.NoneType = None, 
        ):

        if hand is None: hand = self.partial_deal.copy()
        if constraint is None: constraint = self.constraint.shape_constraint
        if constraint is None or constraint.no_shape_cond: return self._random(hand)

        # first satisfy min shape constraints
        for suit in range(4):
            if constraint.suit_min[suit] == 0: continue
            available_cards = np.where(~hand.fixed[suit,:])[0]
            if len(available_cards) < constraint.suit_min[suit]:
                raise ValueError('Given hand conflicts with shape constraints.')
            chosen_cards = np.random.choice(
                available_cards,
                size=constraint.suit_min[suit] - hand.array_rep[constraint.hand, suit,:].sum(), 
                replace=False)
            hand.array_rep[constraint.hand, suit, chosen_cards] = True

        # then satisfy max shape constraints by dealing cards to other three hands
        for suit in range(4):
            if constraint.suit_max[suit] == 13: continue
            available_cards = np.where(~hand.array_rep[:,suit,:].any(axis=0))[0]
            np.random.shuffle(available_cards)
            cards_to_deal = len(available_cards) - constraint.suit_max[suit] + hand.array_rep[constraint.hand, suit,:].sum()
            for i in range(cards_to_deal):
                weights = 13 - hand.array_rep.sum(axis = (1,2)); weights[constraint.hand] = 0
                hand.array_rep[np.random.choice(4, p = weights / weights.sum()), suit, available_cards[i]] = True

        # then deal all remaining cards
        hand = self._random(hand)
        return hand
    
    def _deal_hcp_fixed_shape(self, hand:Hand, constraint: HCPConstraint | types.NoneType = None):
        if not hand._dealt(): raise ValueError('Hand must be fully dealt before calling _deal_hcp_fixed_shape.')
        if constraint is None: constraint = self.constraint.hcp_constraint
        if constraint is None or constraint.no_hcp_cond: return hand

        #region dealing with heuristic weights
        shape = hand._get_shape()
        hand.reset()
        fixed_shape = hand._get_shape()
        cards_to_deal = shape - fixed_shape

        from weights import heuristic_rank_weight
        weights = []
        for idx in range(4):
            if constraint.hcp_max[idx] == 37 and constraint.hcp_min[idx] == 0: weights.append(None); continue
            weights.append(heuristic_rank_weight(constraint.hcp_max[idx], constraint.hcp_min[idx]))
        
        for _ in range(10000): # try 10,000 times to get a valid deal
            for idx in range(4):
                if weights[idx] is None: continue
                for suit in range(4):
                    if cards_to_deal[idx, suit] == 0: continue
                    suit_available = np.where(hand.array_rep[:,suit,:].sum(axis=0) == 0)[0]
                    weight_available = weights[idx][suit_available]
                    hand.array_rep[idx, suit, np.random.choice(
                        suit_available, 
                        size = cards_to_deal[idx, suit], 
                        replace = False, 
                        p = weight_available / weight_available.sum()
                    )] = True
                hand = self._random(hand)
            if constraint.check(hand): return hand
            hand.reset()
        raise ValueError('Failed to deal a hand satisfying the HCP constraints.')
        #endregion
    
    def deal(self, n:int = 50000):
        if n <= 0: raise ValueError('n must be a positive integer.')
        start = perf_counter()
        deals = []
        for _ in tqdm(range(n), desc = 'Dealing hands'):
            hand = self.partial_deal.copy()
            hand = self.constraint.permute_suits.permute(hand)
            if isinstance(self.constraint.shape_constraint, SingleHandShapeConstraint):
                hand = self._deal_shape_single_hand_constraint(hand, self.constraint.shape_constraint)
            else:
                raise NotImplementedError('Dealing with multi-hand shape constraints is not yet supported.')
            hand = self._deal_hcp_fixed_shape(hand, self.constraint.hcp_constraint)
            deals.append(hand)
        end = perf_counter()
        print(f'Dealt {n} hands in {end-start:.2f} seconds ({n/(end-start):.2f} hands/second).')
        return deals

class Simulator():
    def __init__(self, constraint = None, given = None, **kwargs):
        self.dealer = Dealer(constraint, given, **kwargs)
        self.deals = []
        self.n_deals = 0

    def deal(self, n:int = 50000):
        self.deals = self.dealer.deal(n)
        self.n_deals = n
        return self

    def check(self, constraint):
        passed = [constraint.check(deal) for deal in self.deals]
        print(f'{sum(passed)}/{self.n_deals}({sum(passed)/self.n_deals:.2f}) deals passed the constraint check.')
        return sum(passed)/self.n_deals

    def expectation(self):
        return np.mean([deal.array_rep for deal in self.deals], axis=0)


# Example usage
test = Constraint(
    SingleHandShapeConstraint(0, 5, 2, False), 
    HCPConstraint([13,37,37,37],[10,0,0,0]))
dealer = Simulator(constraint = test).deal(50000)
dealer.check(Constraint(None,HCPConstraint([37,37,37,37],[0,15,0,0])))