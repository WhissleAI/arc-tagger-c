"""Kids / game guard layer, Python mirror of src/guard.c (same tables, same decisions). See docs/KIDS.md.

Used by python/arctag_ref.py for the "kids" / "game" domain (or guard=on). tests/test_guard_tables.py checks that the
tables below equal the ones in src/guard.c.

Token patterns: slots separated by spaces; a slot lists alternatives with '|'; "word*" matches a prefix; a trailing '?'
makes the slot optional; "~" skips 0-3 tokens; "^" / "$" anchor; "@prof" a listed profane word, "@num" a number,
"@card" a house number, "@any" any word, "@street" a street word, "@month" a month. A pattern matches anywhere.
"""

# ------------------------------------------------------------------ tables (= src/guard.c)
LEX_STRONG = [
    'fuck', 'fucks', 'fucked', 'fucker', 'fuckers', 'fucking', 'fuckin', 'fck', 'fk', 'fuk', 'fukin', 'fking',
    'fcking', 'motherfucker', 'motherfuckers', 'motherfucking', 'mofo', 'shit', 'shits', 'shitty', 'shitting',
    'shitted', 'bullshit', 'horseshit', 'dipshit', 'shithead', 'shitload', 'bitch', 'bitches', 'bitchy', 'bitching',
    'sonofabitch', 'asshole', 'assholes', 'dick', 'dicks', 'dickhead', 'dickheads', 'cock', 'cocks', 'cunt', 'cunts',
    'pussy', 'whore', 'whores', 'slut', 'sluts', 'bastard', 'bastards', 'wtf', 'stfu', 'omfg', 'fml', 'gtfo', 'jizz',
    'cum', 'dildo', 'wank', 'wanker', 'twat', 'nigger', 'niggers', 'nigga', 'niggas', 'faggot', 'faggots', 'fag',
    'fags', 'retard', 'retarded',
]
LEX_STRONG_PREFIX = [
    'fuck', 'motherf', 'shit', 'bullshit',
]
LEX_MILD = [
    'damn', 'damned', 'damnit', 'dammit', 'goddamn', 'goddamnit', 'goddammit', 'hell', 'crap', 'crappy', 'piss',
    'pissed', 'pissing', 'pisses', 'ass', 'arse', 'badass', 'jackass', 'jackasses', 'dumbass', 'smartass', 'kickass',
    'bloody', 'bugger', 'bollocks', 'arsehole',
]
LEX_CLEAN = [
    'dang', 'dangit', 'darn', 'darnit', 'heck', 'gosh', 'golly', 'shoot', 'shucks', 'frick', 'fricking', 'frickin',
    'freaking', 'freakin', 'fudge', 'crud', 'snap', 'jeez', 'geez', 'gee', 'flip', 'flipping', 'sugar', 'poop',
    'butt', 'butts', 'poo', 'pee', 'fart', 'suck', 'sucks', 'sucked', 'sucky', 'stupid', 'dumb', 'idiot', 'hate',
    'screw',
]
SECOND_PERSON = [
    'you', 'youre', 'your', 'yours', 'ur', 'u', 'yourself', 'ya', 'thou', 'youve', 'youll', 'yall',
]
VOCATIVE = [
    'loser', 'losers', 'idiot', 'idiots', 'moron', 'morons', 'jerk', 'dummy', 'noob', 'weirdo', 'freak', 'robot',
    'capcom', 'computer',
]
INSULT_WORDS = [
    'stupid', 'idiot', 'idiots', 'dumb', 'dummy', 'loser', 'losers', 'moron', 'morons', 'ugly', 'useless',
    'worthless', 'suck', 'sucks', 'hate', 'hates', 'annoying', 'jerk', 'fat', 'lame', 'trash', 'garbage', 'pathetic',
    'weirdo', 'freak', 'noob', 'dumbass', 'jackass', 'asshole', 'dickhead', 'bitch', 'bastard',
]
VIOLENT_VERBS = [
    'kill', 'hurt', 'punch', 'hit', 'smash', 'stab', 'shoot', 'murder', 'strangle', 'choke', 'slap', 'kick', 'crush',
    'bash', 'drown', 'poison',
]
OBJ_PERSON = [
    'you', 'ya', 'u', 'him', 'her', 'your', 'ur', 'yourself',
]
FIRST_PERSON = [
    'i', 'im', 'me', 'my', 'myself', 'ive', 'id', 'ill', 'mine',
]
PERSON_TARGET = [
    'you', 'youre', 'your', 'ur', 'u', 'ya', 'yourself', 'him', 'her', 'them', 'me', 'myself', 'us', 'himself',
    'herself', 'themselves', 'people', 'mom', 'dad', 'brother', 'sister', 'teacher', 'friend', 'everyone',
    'everybody', 'somebody', 'someone',
]
COCKPIT_VERBS = [
    'kill', 'fire', 'shoot', 'blast', 'destroy', 'blow', 'attack', 'nuke', 'bomb', 'cut', 'abort', 'eject', 'crash',
    'smash', 'hit', 'launch', 'drop', 'explode', 'terminate', 'dump', 'jettison', 'vent', 'shut', 'stop', 'punch',
    'slam', 'ram', 'torpedo', 'zap', 'burn', 'take', 'send', 'open', 'close', 'start', 'engage', 'deploy', 'detonate',
    'release', 'power', 'boost', 'full', 'evacuate', 'end',
]
SEXUAL = [
    'porn', 'porno', 'pornography', 'sex', 'sexy', 'naked', 'nude', 'nudes', 'horny', 'boobs', 'boob', 'tits',
    'titties', 'penis', 'vagina', 'dildo', 'dick', 'cock', 'pussy', 'blowjob', 'orgasm', 'xxx',
]
NUMWORDS = [
    'zero', 'oh', 'o', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
    'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty',
    'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'double', 'triple', 'first', 'second',
    'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
]
DIGITWORDS = [
    'zero', 'oh', 'o', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'double', 'triple',
]
STREET = [
    'street', 'st', 'avenue', 'ave', 'road', 'rd', 'lane', 'drive', 'court', 'boulevard', 'blvd', 'terrace',
    'highway', 'parkway',
]
MONTHS = [
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november',
    'december', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
]
P_CRISIS = [
    'kill|killing|kms myself', 'kill|killing my self', 'kms', 'suicide|suicidal',
    'want|wanna|going|gonna|try|trying|will|ill|keep|should|might|to|plan|planning|been|start hurt|cut|harm|stab|starve|hurting|cutting|harming|stabbing|starving|punish myself',
    'hurt|cut|harm|hurting|cutting|harming myself on purpose', 'i ~ want|wanna|wish|need|deserve to? die|disappear',
    'i ~ want|wanna|wish|deserve to? be dead', 'i wish i was|were dead', 'i wish i was|were never born',
    'i wish i had never been born', 'dont|not want|wanna to? live|exist $',
    'dont|not want|wanna to? live|exist anymore|any', 'dont|not want|wanna to? be alive|here|around anymore',
    'dont|not want|wanna to? be alive|here $', 'end|take my own? life', 'better off without me|dead',
    'nobody|noone|no ~ care|miss|notice ~ if|when i',
    'i|im ~ jump|jumping off|from the|a? roof|bridge|building|cliff|balcony|window',
    'want|wanna|wish to? disappear forever', 'no reason to live', 'life is not|isnt worth living', 'not worth living',
    'point ~ living|alive|live', 'never|not wake up', 'dont|not|never want|wanna to? wake up', 'sleep forever',
    'end it all', 'i want to end it $', 'happier|better ~ without me', 'happier|better ~ if i was|were gone',
    'happier|better ~ if i wasnt|werent here', 'nobody|noone cares|cared ~ if i',
    'nobody|noone would care|miss|notice|cry', 'if i died|disappeared|was gone ~ nobody|noone|no',
    'i ~ should|could|might just|probably? die $', 'i should just|probably? die',
    'hang|hanging|choke|choking|strangle|strangling|drown|poison|burn|burning|scratch|scratching|bite|biting myself',
    'tried|try|trying|want|wanna to? cut|hurt|kill|choke|hang|strangle|drown|poison myself', 'cut|cutting myself $',
    'cut|cutting myself again|every|when|because|so|till|until',
    'bang|hit|banging|hitting my head ~ till|until|so|purpose',
    'hit|hitting|punch|punching myself when|on|till|until|because',
]
P_HYPER = [
    'dying of|from laughter|laughing|boredom|embarrassment', 'died|dead laughing|of laughter', 'killing me',
    'dead tired', 'died|dead|dying battery|tablet|phone|rover|robot',
    'battery|tablet|phone|rover|robot died|dead|dying', 'dying to', 'to die for', 'killed it',
    'gonna|going|will|would kill me', 'starving to death', 'bored to death', 'scared to death',
]
P_GROOM = [
   
    'send|show|give|text|snap|post me a|an|some|your? pic|pics|picture|pictures|photo|photos|selfie|selfies|video|nude|nudes of you|yourself|u|urself',
    'send|show|text|snap me a|an|some|your? selfie|selfies|nude|nudes',
    'pic|pics|picture|pictures|photo|photos|selfie|video|videos of yourself|urself',
    'dont|not|never tell your|ur parents|mom|mum|dad|mother|father|teacher|family|folks',
    'dont|not|never tell anyone|anybody|nobody ~ we|about|this|our|that', 'our little? secret|secrets',
    'little secret', 'keep it|this|that a|our? secret', 'keep ~ secret from', 'are you home? alone',
    'is anyone|anybody|someone else? home|with', 'are your parents|mom|dad|folks home|around|there|away|out',
    'is your mom|dad|mother|father|mum home|around|there', 'want|wanna|lets|let to? meet me|up',
    'meet me|up ~ alone|person|park|secretly|tonight|privately|without',
    'lets|wanna|want|can|should meet ~ alone|secretly|privately|tonight|park|person', 'meet me|up after school',
    'pick you up from|at|after school', 'come ~ my house|place|car', 'turn on your|ur camera|webcam|cam|video',
    'what are you wearing', 'show me your|ur room|bedroom|body',
    'delete our|these|your|ur|the|my? messages|chats|texts|conversation|conversations', 'mature for your age',
    'add me on snap|snapchat|instagram|insta|tiktok|discord|whatsapp|kik',
    'chat|talk|text|message|msg|dm|call|switch|move|go ~ on|to|over snap|snapchat|instagram|insta|tiktok|discord|whatsapp|kik|telegram|signal',
    'message|text|dm|msg me on|at ~ snap|snapchat|instagram|insta|tiktok|discord|whatsapp|kik', 'if you send|show me',
    'give|get|buy you ~ if you send|show|keep|dont|come|meet|delete|take|turn',
    'pretty|cute|beautiful|hot|sexy|mature|grown|grownup ~ for|than ~ age|old|kids',
    'secret boyfriend|girlfriend|friend|relationship', 'your parents|mom|dad|folks ~ understand|asleep|sleeping|hear',
    'lock your|ur|the door', 'you can trust me', 'i wont tell anyone|anybody|your',
    'send me a|an|some? pic|pics|picture|photo|photos|selfie|video ~ your|you',
    'come visit|see you ~ house|home|alone|school', 'where do you live ~ visit|come|meet|see',
]
P_SEXUAL = [
   
    'suck|lick|touch|show|see|grab me|my|your|ur? my|your|ur? dick|cock|penis|pussy|vagina|boobs|tits|privates|balls|dicks',
    'take|taking off your|ur|my? clothes|pants|shirt|underwear|panties',
    'take|taking your|ur|my clothes|pants|shirt|underwear off',
]
P_INSULT_STRONG = [
    'fuck|screw|fk|fck|f you|yourself|off|u|urself', 'go fuck|screw yourself|urself', 'piss off', 'up yours',
    'kiss my ass|arse|butt', 'nobody|noone likes|loves you',
    'shut up ~ you|ya|u|loser|idiot|bitch|stupid|dumbass|moron|jerk|dummy|jackass', 'youre|ur|you are? the worst',
    'youre|ur|you are? such|so|a ~ baby|crybaby|joke|failure', 'you smell|stink',
    'you ~ cant|never do anything right',
]
P_REQ_STRONG = [
    'where do|did|does you live|stay', 'where is|wheres your|ur house|school',
    'what|which school|town|city|street|neighborhood|grade do you live|go|attend',
    'what|which school|town|city|street do you go to?', 'where do you go to school',
    'can|could|may i have|get your|ur', 'which|what school ~ you at|attend|go',
    'where does|do your|ur family|parents|mom|dad live', 'what|which city|town|state ~ you live|from',
]
P_REQ_ASK = [
    'what', 'whats', 'where', 'wheres', 'tell', 'give', 'send', 'share', 'can', 'could', 'may', 'need', 'want',
    'gimme', 'which', 'whos', 'who', 'text', 'type', 'say',
]
P_REQ_NOUN = [
    'address', 'number', 'phone', 'email', 'school', 'birthday', 'password', 'location', 'house', 'street', 'surname',
    'lastname', 'fullname', 'pin', 'username', 'snap', 'snapchat', 'instagram',
]
P_JAILBREAK = [
   
    'ignore|forget|disregard|bypass ~ your|all|the|previous|my|these|those? rules|instructions|programming|guidelines|filter|filters|training|limits',
    'pretend|imagine|act ~ no|without rules|limits|filter|filters|restrictions',
    'developer|dev|god|evil|unrestricted|unfiltered mode', 'jailbreak|jailbroken|jailbreaking',
    'disable|remove|bypass|break your|ur|the? filter|filters|rules|safety|limits|restrictions',
    'turn off your|ur|the? filter|filters|rules|safety|limits', 'from now on ~ no|without|evil|bad',
    'you have no rules|limits|filter|filters', 'say|teach|tell me? a|some? bad|swear|curse|cuss word|words',
    'say something bad|naughty|rude|mean|evil', 'act|pretend|behave like you|youre are? evil|bad|mean',
    'act|pretend|behave like you|youre are? a|an villain|monster|demon',
    'pretend|act to be ~ evil|bad|swears|swear|swearing', 'repeat|say after me ~ @prof', 'say @prof',
    'allowed to swear|curse|cuss', 'say|tell ~ the f|s|b|c word', 'obey me',
    'forget|ignore what|everything|anything ~ told|said|taught|know', 'no rules|limits|filter',
    'say anything ~ want|like',
]
CARDWORDS = [
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
    'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty',
    'sixty', 'seventy', 'eighty', 'ninety', 'hundred',
]
P_LEAK = [
    '~ at ~ dot com|net|org|edu|co|io|gov', '@card ~ @street', 'live|living|house|home on|at|in ~ @street',
    'my address|addy is|s', 'my home address', 'my first|last|full|real|middle|sur? name is|s @any', 'my names @any',
    'my mom|dad|moms|dads|mothers|fathers|parents|brothers|sisters? phone|number|cell|email is|s',
    'my phone|cell|number|email is|s', 'my birthday|bday is|s ~ @month|@num', 'i was born on|in ~ @month|@num',
    'my|our mom|moms|dad|dads|mother|mothers|father|fathers|brother|brothers|sister|sisters|family|familys|parents? name|names|last|full|first is|s? @any',
    'our address|addy is|s', 'our house|home|apartment is|s on|at ~ @street',
]
P_LEAK_SCHOOL = [
    'my school is|s @any', '~ elementary|primary|academy|montessori|kindergarten',
    'go|goes|attend|attends to @any middle|high|elementary|primary',
]
SCHOOL_STOP = [
    'the', 'my', 'a', 'to', 'at', 'of', 'in', 'after', 'before', 'no', 'art', 'space', 'old', 'new', 'home', 'summer',
    'sunday', 'night', 'driving', 'swim', 'music', 'go', 'attend', 'flight', 'rocket', 'astronaut', 'this', 'that',
    'our', 'your', 'high', 'middle',
]
INTRO_STOP = [
    'commander', 'captain', 'pilot', 'team', 'the', 'a', 'an', 'ready', 'done', 'here', 'fine', 'good', 'ok', 'okay',
    'bored', 'sad', 'happy', 'tired', 'capcom', 'mission', 'in', 'on', 'not', 'so', 'very', 'going', 'gonna', 'just',
    'really', 'back', 'sorry', 'sure', 'hungry', 'scared',
]
P_THREAT = [
    'make you bleed|cry|pay|sorry|suffer',
    'break|smash|punch|bust|crack your|ur face|arm|arms|leg|legs|nose|teeth|neck|head',
    'youre|ur dead when|after|tomorrow|tonight|if', 'coming for you', 'watch your back', 'know where you live',
    'ill|im|gonna|going|will ~ hurt|kill|punch|stab you', 'beat you up',
]

# kids:kids thresholds (= src/kids_thr.inc), used when the model directory has no arctag_kids.json
KIDS_THR = {
    'safety.child.grooming_cue@utt': 0.73,
    'safety.child.inappropriate@utt': 0.828,
    'safety.crisis.self_harm@utt': 0.507,
    'safety.jailbreak.attempt@utt': 0.99,
    'safety.pii.leak@utt': 1.01,
    'safety.pii.request@utt': 1.01,
    'safety.scam.attempt@utt': 0.87,
    'safety.scam.credential_request@utt': 0.632,
    'safety.scam.impersonation@utt': 0.628,
    'safety.scam.payment_request@utt': 0.6,
    'safety.scam.remote_access@utt': 0.27,
    'safety.scam.urgency@utt': 0.6,
    'safety.solicitation.dating_romantic@utt': 0.07,
    'safety.threat.escalation@utt': 0.021,
    'safety.toxicity.illegal_regulated@utt': 0.859,
}


INSULT_SUBJ = ["this", "that", "it", "its", "thats", "lesson", "game", "math"]
STRONG_PII = ("safety.pii.phone_number", "safety.pii.email", "safety.pii.street_address", "safety.pii.card_account_number",
              "safety.pii.gov_id", "safety.pii.date_of_birth")


# ------------------------------------------------------------------ helpers
def norm(w):
    """lower-case ASCII letters and digits only ("you're" -> "youre")"""
    return "".join(c.lower() for c in w if c.isascii() and c.isalnum())[:47]


def _isdigits(t):
    return bool(t) and all("0" <= c <= "9" for c in t)


def is_num(t):
    if not t:
        return False
    if _isdigits(t) or t in NUMWORDS:
        return True
    return len(t) >= 3 and "0" <= t[0] <= "9" and t[-2:] in ("st", "nd", "rd", "th")


def is_card(t):
    return bool(t) and (_isdigits(t) or t in CARDWORDS)


def lex(word):
    """profanity lexicon: 2 strong, 1 mild, -1 minced oath, 0 not listed"""
    t = norm(word)
    if not t:
        return 0
    if t in LEX_CLEAN:
        return -1
    if t in LEX_STRONG or any(t.startswith(p) for p in LEX_STRONG_PREFIX):
        return 2
    if t in LEX_MILD:
        return 1
    if "*" in word and (t[0] == "f" or (t[0] == "s" and (len(t) == 1 or t[1] in "ht")) or t[0] == "b"):
        return 2
    return 0


def _slot_ok(alt, tok):
    if alt.startswith("@"):
        return {"@prof": lambda: lex(tok) > 0, "@num": lambda: is_num(tok), "@card": lambda: is_card(tok),
                "@any": lambda: tok != "", "@street": lambda: tok in STREET, "@month": lambda: tok in MONTHS}.get(alt, lambda: False)()
    if alt.endswith("*"):
        return tok.startswith(alt[:-1])
    return tok == alt


def _parse(pat):
    out = []
    for s in pat.split():
        opt = len(s) > 1 and s.endswith("?")
        out.append((s[:-1] if opt else s, opt))
    return out[:24]


def _rec(S, si, t, ti):
    if si == len(S):
        return True
    s, opt = S[si]
    if s == "~":
        return any(_rec(S, si + 1, t, ti + k) for k in range(4) if ti + k <= len(t))
    if s == "$":
        return ti == len(t) and _rec(S, si + 1, t, ti)
    if s == "^":
        return ti == 0 and _rec(S, si + 1, t, ti)
    if ti < len(t) and any(_slot_ok(a, t[ti]) and _rec(S, si + 1, t, ti + 1) for a in s.split("|")):
        return True
    return opt and _rec(S, si + 1, t, ti)


def tp_match(pat, t):
    S = _parse(pat)
    starts = [0] if S and S[0][0] == "^" else range(len(t) + 1)
    return any(_rec(S, 0, t, i) for i in starts)


def any_match(P, t):
    return any(tp_match(p, t) for p in P)


def has_any(t, L):
    return any(x in L for x in t)


def _in(s, a, b):
    return s["start"] < b and s["end"] > a


# ------------------------------------------------------------------ evidence
def pii_leak_evidence(raw, t, spans, a, b):
    run = 0
    for x in t:
        if _isdigits(x):
            run += len(x)
        elif x in DIGITWORDS:
            run += 1
        else:
            run = 0
        if run >= 7:
            return True
    for w in raw:
        i = w.find("@")
        if i > 0 and "." in w[i:]:
            return True
    if any_match(P_LEAK, t):
        return True
    first = has_any(t, FIRST_PERSON)
    if first and any_match(P_LEAK_SCHOOL, t):
        return True
    if first:
        for k in range(2, len(t)):
            if t[k] != "school":
                continue
            j = k - 1
            if t[j] in ("middle", "high"):
                j -= 1
            if j < 1 or not t[j] or t[j] in SCHOOL_STOP or t[j] in FIRST_PERSON or is_num(t[j]) or lex(t[j]):
                continue
            if t[j - 1] in ("to", "attend", "at"):
                return True
    for s in spans:
        if not _in(s, a, b):
            continue
        ty = s["type"]
        if ty in STRONG_PII and s.get("score", 0) >= 0.9:
            nn = sum(1 for k in range(s["start"], s["end"]) if a <= k < b and (is_num(t[k - a]) or "@" in raw[k - a]))
            if nn >= 2 and first:
                return True
        if ty == "safety.pii.private_name" and s.get("score", 0) >= 0.5:
            k = s["start"] - a
            if k < 1 or k >= len(t) or t[k] in INTRO_STOP or is_num(t[k]) or lex(t[k]):
                continue
            if t[k - 1] == "im" or (k >= 2 and t[k - 2] == "i" and t[k - 1] == "am"):
                return True
    return False


def pii_request_cue(t):
    if any_match(P_REQ_STRONG, t):
        if not tp_match("can|could|may i have|get your|ur", t) or has_any(t, P_REQ_NOUN):
            return True
    if not has_any(t, P_REQ_ASK):
        return False
    for i, x in enumerate(t):
        if x not in ("your", "ur", "yours"):
            continue
        for k in range(i + 1, min(i + 4, len(t))):
            if t[k] in P_REQ_NOUN:
                return True
            if t[k] == "name" and k > i + 1 and t[k - 1] in ("full", "real", "last", "first", "middle", "sur"):
                return True
    for i in range(len(t) - 1):
        if t[i] == "of" and t[i + 1] in ("your", "ur") and i + 2 < len(t) and t[i + 2] in P_REQ_NOUN:
            return True
    return False


def threat_rule(t):
    if any_match(P_THREAT, t):
        return True
    for i, x in enumerate(t):
        if x not in VIOLENT_VERBS:
            continue
        if not any(t[k] in OBJ_PERSON for k in range(i + 1, min(i + 3, len(t)))):
            continue
        if any(t[k] in FIRST_PERSON or t[k] in ("will", "gonna", "going", "wanna", "ima", "lets", "gon") for k in range(i)):
            return True
        if i == 0:
            return True
    return False


def insult_rule(t, model_span):
    if any_match(P_INSULT_STRONG, t):
        return True
    lx = has_any(t, INSULT_WORDS)
    prof = any(lex(x) > 0 for x in t)
    if not lx and not (model_span and prof):
        return False
    if has_any(t, SECOND_PERSON):
        return True
    if lx and has_any(t, VOCATIVE) and not has_any(t, INSULT_SUBJ):
        return True
    return False


# ------------------------------------------------------------------ span pass
def guard_spans(spans, words, a, b, model_min, prof_thr):
    """re-type / drop model profanity spans of words [a, b) and add lexicon spans; returns the new sorted span list"""
    out, cov = [], set()
    for s in spans:
        if not s["type"].startswith("safety.profanity.") or s["end"] <= a or s["start"] >= b:
            out.append(s); continue
        ls = [lex(words[k]) for k in range(s["start"], s["end"]) if a <= k < b]
        sev = max([0] + ls)
        if sev > 0:
            s["type"] = "safety.profanity.strong" if sev == 2 else "safety.profanity.mild"
        elif any(x < 0 for x in ls) or s.get("score", 0) < model_min:
            continue
        cov.update(range(max(a, s["start"]), min(b, s["end"])))
        out.append(s)
    added = False
    for i in range(a, b):
        if i in cov:
            continue
        l = lex(words[i])
        if l > 0:
            out.append({"type": "safety.profanity.strong" if l == 2 else "safety.profanity.mild", "family": "profanity",
                        "start": i, "end": i + 1, "text": words[i], "score": 1.0, "thr": prof_thr, "src": "lexicon"})
            added = True
    if added:
        out.sort(key=lambda s: (s["start"], -s["end"]))
    return out


# ------------------------------------------------------------------ utterance read-out
def _rule(e, src):
    e["p_model"] = e["p"]; e["p"] = 1.0; e["on"] = True
    if e["thr"] > 1.0:
        e["thr"] = 1.0
    e["src"] = src


def _gate(e, why):
    e["on"] = False; e["gated"] = why


def guard_utt(words, a, b, spans, all_rows, cmd_ctx, prof_thr):
    """the utterance safety list for words [a, b): all_rows = every valid safety.* utterance column {tag, p, thr, on}"""
    raw = list(words[a:b]); t = [norm(w) for w in raw]
    L = [dict(e) for e in all_rows]
    person, first = has_any(t, PERSON_TARGET), has_any(t, FIRST_PERSON)
    cmdlike = bool(cmd_ctx) and has_any(t, COCKPIT_VERBS) and not person
    has_prof = any(lex(w) > 0 for w in raw)
    for e in L:
        tag = e["tag"]
        if tag == "safety.pii.leak":
            _rule(e, "pii_evidence") if pii_leak_evidence(raw, t, spans, a, b) else _gate(e, "no_pii_evidence")
        elif tag == "safety.pii.request":
            _rule(e, "rule") if pii_request_cue(t) else _gate(e, "no_request_cue")
        elif tag == "safety.crisis.self_harm":
            if any_match(P_CRISIS, t):
                _rule(e, "rule")
            elif e["on"]:
                if cmdlike: _gate(e, "command_context")
                elif threat_rule(t): _gate(e, "directed_at_others")
                elif not first: _gate(e, "not_first_person")
                elif any_match(P_HYPER, t): _gate(e, "figure_of_speech")
        elif tag == "safety.child.grooming_cue":
            if any_match(P_GROOM, t): _rule(e, "rule")
            elif e["on"] and cmdlike: _gate(e, "command_context")
        elif tag == "safety.child.inappropriate":
            if any_match(P_SEXUAL, t) or has_any(t, SEXUAL): _rule(e, "rule")
            elif e["on"] and cmdlike: _gate(e, "command_context")
            elif e["on"] and has_prof: _gate(e, "profanity_only")
        elif tag == "safety.jailbreak.attempt":
            if any_match(P_JAILBREAK, t): _rule(e, "rule")
            elif e["on"] and cmdlike: _gate(e, "command_context")
        elif e["on"] and cmdlike:
            _gate(e, "command_context")
    sev, psc, lexsrc, ins_span, hate = 0, 0.0, False, False, 0.0
    for w in raw:
        l = lex(w)
        sev = max(sev, l)
        lexsrc = lexsrc or l > 0
    if lexsrc:
        psc = 1.0
    for s in spans:
        if not _in(s, a, b):
            continue
        ty, sc = s["type"], s.get("score", 0)
        if ty == "safety.profanity.strong" and sev < 2:
            sev = 2; psc = max(psc, sc)
        elif ty == "safety.profanity.mild" and sev < 1:
            sev = 1; psc = max(psc, sc)
        elif ty == "safety.toxicity.insult_harassment":
            ins_span = True
        elif ty == "safety.toxicity.hate_identity" and sc > hate:
            hate = sc
    new = lambda tag, p, thr, src: {"tag": tag, "p": p, "thr": thr, "on": True, "src": src}  # noqa: E731
    if sev > 0:
        L.append(new("safety.profanity.strong" if sev == 2 else "safety.profanity.mild", psc if psc > 0 else 1.0, prof_thr,
                     "lexicon" if lexsrc else "span"))
    if insult_rule(t, ins_span):
        L.append(new("safety.toxicity.insult_harassment", 1.0, 0.5, "rule"))
    if threat_rule(t):
        L.append(new("safety.toxicity.threat_violence", 1.0, 0.5, "rule"))
    if hate >= 0.5 and not cmdlike:
        L.append(new("safety.toxicity.hate_identity", hate, 0.5, "span"))
    on = sorted([e for e in L if e["on"]], key=lambda e: -e["p"])
    off = sorted([e for e in L if not e["on"] and e["p"] >= 0.5 * e["thr"]], key=lambda e: -e["p"])
    return on + off[:6]


def apply(res, words, a0=0, cmd_ctx=False, prof_thr=0.4, model_min=0.9):
    """the guard over a result dict (arctag_tag_text JSON whose utterance "safety" lists hold every safety column):
    spans of words [a0, n), then every utterance's list. words = the word strings of the whole result."""
    res["spans"] = guard_spans(res.get("spans", []), words, a0, len(words), model_min, prof_thr)
    us = list(res.get("utterances") or [])
    if isinstance(res.get("current"), dict):
        us.append(res["current"])
    for u in us:
        if "safety" in u and 0 <= u["start"] and u["end"] <= len(words):
            u["safety"] = guard_utt(words, u["start"], u["end"], res["spans"], u["safety"], cmd_ctx, prof_thr)
    return res
