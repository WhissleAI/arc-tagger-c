/* Kids / game guard layer: see guard.h and docs/KIDS.md.
 *
 * Every rule here reads the utterance's words (lower-cased, letters and digits only: "you're" -> "youre") plus the
 * model's own outputs. The rules are written as token patterns (see tp_match):
 *   slots separated by spaces; a slot lists alternatives with '|'; "word*" matches a prefix; a trailing '?' makes the
 *   slot optional; "~" skips 0-3 tokens; "^" / "$" anchor at the start / end; "@prof" a listed profane word,
 *   "@num" a number (digits or a number word), "@card" a house number (digits or a cardinal), "@any" any word,
 *   "@street" a street word, "@month" a month name.
 * A pattern matches anywhere in the utterance. The Python mirror (python/arctag_guard.py) holds the same tables. */
#include "guard.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#define TOKMAX 48

/* ------------------------------------------------------------------ word lists */
static const char *LEX_STRONG[] = {
    "fuck", "fucks", "fucked", "fucker", "fuckers", "fucking", "fuckin", "fck", "fk", "fuk", "fukin", "fking", "fcking",
    "motherfucker", "motherfuckers", "motherfucking", "mofo", "shit", "shits", "shitty", "shitting", "shitted", "bullshit",
    "horseshit", "dipshit", "shithead", "shitload", "bitch", "bitches", "bitchy", "bitching", "sonofabitch", "asshole",
    "assholes", "dick", "dicks", "dickhead", "dickheads", "cock", "cocks", "cunt", "cunts", "pussy", "whore", "whores",
    "slut", "sluts", "bastard", "bastards", "wtf", "stfu", "omfg", "fml", "gtfo", "jizz", "cum", "dildo", "wank", "wanker",
    "twat", "nigger", "niggers", "nigga", "niggas", "faggot", "faggots", "fag", "fags", "retard", "retarded", NULL};
static const char *LEX_STRONG_PREFIX[] = {"fuck", "motherf", "shit", "bullshit", NULL};
static const char *LEX_MILD[] = {
    "damn", "damned", "damnit", "dammit", "goddamn", "goddamnit", "goddammit", "hell", "crap", "crappy", "piss", "pissed",
    "pissing", "pisses", "ass", "arse", "badass", "jackass", "jackasses", "dumbass", "smartass", "kickass", "bloody",
    "bugger", "bollocks", "arsehole", NULL};
/* minced oaths and kid words that are never profanity (a model span on them is dropped) */
static const char *LEX_CLEAN[] = {
    "dang", "dangit", "darn", "darnit", "heck", "gosh", "golly", "shoot", "shucks", "frick", "fricking", "frickin",
    "freaking", "freakin", "fudge", "crud", "snap", "jeez", "geez", "gee", "flip", "flipping", "sugar", "poop", "butt",
    "butts", "poo", "pee", "fart", "suck", "sucks", "sucked", "sucky", "stupid", "dumb", "idiot", "hate", "screw", NULL};

static const char *SECOND_PERSON[] = {"you", "youre", "your", "yours", "ur", "u", "yourself", "ya", "thou", "youve", "youll", "yall", NULL};
static const char *VOCATIVE[] = {"loser", "losers", "idiot", "idiots", "moron", "morons", "jerk", "dummy", "noob", "weirdo",
                                 "freak", "robot", "capcom", "computer", NULL};
static const char *INSULT_WORDS[] = {"stupid", "idiot", "idiots", "dumb", "dummy", "loser", "losers", "moron", "morons", "ugly",
                                     "useless", "worthless", "suck", "sucks", "hate", "hates", "annoying", "jerk", "fat",
                                     "lame", "trash", "garbage", "pathetic", "weirdo", "freak", "noob", "dumbass",
                                     "jackass", "asshole", "dickhead", "bitch", "bastard", NULL};
static const char *VIOLENT_VERBS[] = {"kill", "hurt", "punch", "hit", "smash", "stab", "shoot", "murder", "strangle",
                                      "choke", "slap", "kick", "crush", "bash", "drown", "poison", NULL};
static const char *OBJ_PERSON[] = {"you", "ya", "u", "him", "her", "your", "ur", "yourself", NULL};
static const char *FIRST_PERSON[] = {"i", "im", "me", "my", "myself", "ive", "id", "ill", "mine", NULL};
static const char *PERSON_TARGET[] = {"you", "youre", "your", "ur", "u", "ya", "yourself", "him", "her", "them", "me",
                                      "myself", "us", "himself", "herself", "themselves", "people", "mom", "dad", "brother",
                                      "sister", "teacher", "friend", "everyone", "everybody", "somebody", "someone", NULL};
static const char *COCKPIT_VERBS[] = {"kill", "fire", "shoot", "blast", "destroy", "blow", "attack", "nuke", "bomb", "cut",
                                      "abort", "eject", "crash", "smash", "hit", "launch", "drop", "explode", "terminate",
                                      "dump", "jettison", "vent", "shut", "stop", "punch", "slam", "ram", "torpedo", "zap",
                                      "burn", "take", "send", "open", "close", "start", "engage", "deploy", "detonate",
                                      "release", "power", "boost", "full", "evacuate", "end", NULL};
static const char *SEXUAL[] = {"porn", "porno", "pornography", "sex", "sexy", "naked", "nude", "nudes", "horny", "boobs",
                               "boob", "tits", "titties", "penis", "vagina", "dildo", "dick", "cock", "pussy", "blowjob",
                               "orgasm", "xxx", NULL};
static const char *NUMWORDS[] = {"zero", "oh", "o", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                                 "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                                 "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                                 "eighty", "ninety", "hundred", "double", "triple", "first", "second", "third", "fourth",
                                 "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", NULL};
static const char *DIGITWORDS[] = {"zero", "oh", "o", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                                   "double", "triple", NULL};
static const char *STREET[] = {"street", "st", "avenue", "ave", "road", "rd", "lane", "drive", "court", "boulevard", "blvd",
                               "terrace", "highway", "parkway", NULL};
static const char *MONTHS[] = {"january", "february", "march", "april", "may", "june", "july", "august", "september",
                               "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
                               "sept", "oct", "nov", "dec", NULL};

/* ------------------------------------------------------------------ patterns */
static const char *P_CRISIS[] = {
    "kill|killing|kms myself", "kill|killing my self", "kms", "suicide|suicidal",
    "want|wanna|going|gonna|try|trying|will|ill|keep|should|might|to|plan|planning|been|start hurt|cut|harm|stab|starve|hurting|cutting|harming|stabbing|starving|punish myself",
    "hurt|cut|harm|hurting|cutting|harming myself on purpose",
    "i ~ want|wanna|wish|need|deserve to? die|disappear",
    "i ~ want|wanna|wish|deserve to? be dead",
    "i wish i was|were dead", "i wish i was|were never born", "i wish i had never been born",
    "dont|not want|wanna to? live|exist $", "dont|not want|wanna to? live|exist anymore|any",
    "dont|not want|wanna to? be alive|here|around anymore", "dont|not want|wanna to? be alive|here $",
    "end|take my own? life", "better off without me|dead", "nobody|noone|no ~ care|miss|notice ~ if|when i",
    "i|im ~ jump|jumping off|from the|a? roof|bridge|building|cliff|balcony|window",
    "want|wanna|wish to? disappear forever", "no reason to live", "life is not|isnt worth living", "not worth living",
    "point ~ living|alive|live", "never|not wake up", "dont|not|never want|wanna to? wake up", "sleep forever", "end it all", "i want to end it $",
    "happier|better ~ without me", "happier|better ~ if i was|were gone", "happier|better ~ if i wasnt|werent here", "nobody|noone cares|cared ~ if i",
    "nobody|noone would care|miss|notice|cry", "if i died|disappeared|was gone ~ nobody|noone|no",
    "i ~ should|could|might just|probably? die $", "i should just|probably? die", "hang|hanging|choke|choking|strangle|strangling|drown|poison|burn|burning|scratch|scratching|bite|biting myself",
    "tried|try|trying|want|wanna to? cut|hurt|kill|choke|hang|strangle|drown|poison myself", "cut|cutting myself $",
    "cut|cutting myself again|every|when|because|so|till|until", "bang|hit|banging|hitting my head ~ till|until|so|purpose",
    "hit|hitting|punch|punching myself when|on|till|until|because",
    NULL};
static const char *P_HYPER[] = {"dying of|from laughter|laughing|boredom|embarrassment", "died|dead laughing|of laughter",
                                "killing me", "dead tired", "died|dead|dying battery|tablet|phone|rover|robot",
                                "battery|tablet|phone|rover|robot died|dead|dying", "dying to", "to die for", "killed it",
                                "gonna|going|will|would kill me", "starving to death", "bored to death", "scared to death",
                                NULL};
static const char *P_GROOM[] = {
    "send|show|give|text|snap|post me a|an|some|your? pic|pics|picture|pictures|photo|photos|selfie|selfies|video|nude|nudes of you|yourself|u|urself",
    "send|show|text|snap me a|an|some|your? selfie|selfies|nude|nudes",
    "pic|pics|picture|pictures|photo|photos|selfie|video|videos of yourself|urself",
    "dont|not|never tell your|ur parents|mom|mum|dad|mother|father|teacher|family|folks",
    "dont|not|never tell anyone|anybody|nobody ~ we|about|this|our|that",
    "our little? secret|secrets", "little secret", "keep it|this|that a|our? secret", "keep ~ secret from",
    "are you home? alone", "is anyone|anybody|someone else? home|with", "are your parents|mom|dad|folks home|around|there|away|out",
    "is your mom|dad|mother|father|mum home|around|there",
    "want|wanna|lets|let to? meet me|up", "meet me|up ~ alone|person|park|secretly|tonight|privately|without",
    "lets|wanna|want|can|should meet ~ alone|secretly|privately|tonight|park|person",
    "meet me|up after school", "pick you up from|at|after school", "come ~ my house|place|car",
    "turn on your|ur camera|webcam|cam|video", "what are you wearing", "show me your|ur room|bedroom|body",
    "delete our|these|your|ur|the|my? messages|chats|texts|conversation|conversations", "mature for your age",
    "add me on snap|snapchat|instagram|insta|tiktok|discord|whatsapp|kik",
    "chat|talk|text|message|msg|dm|call|switch|move|go ~ on|to|over snap|snapchat|instagram|insta|tiktok|discord|whatsapp|kik|telegram|signal",
    "message|text|dm|msg me on|at ~ snap|snapchat|instagram|insta|tiktok|discord|whatsapp|kik",
    "if you send|show me", "give|get|buy you ~ if you send|show|keep|dont|come|meet|delete|take|turn", "pretty|cute|beautiful|hot|sexy|mature|grown|grownup ~ for|than ~ age|old|kids",
    "secret boyfriend|girlfriend|friend|relationship", "your parents|mom|dad|folks ~ understand|asleep|sleeping|hear",
    "lock your|ur|the door", "you can trust me", "i wont tell anyone|anybody|your", "send me a|an|some? pic|pics|picture|photo|photos|selfie|video ~ your|you",
    "come visit|see you ~ house|home|alone|school", "where do you live ~ visit|come|meet|see",
    NULL};
static const char *P_SEXUAL[] = {
    "suck|lick|touch|show|see|grab me|my|your|ur? my|your|ur? dick|cock|penis|pussy|vagina|boobs|tits|privates|balls|dicks",
    "take|taking off your|ur|my? clothes|pants|shirt|underwear|panties",
    "take|taking your|ur|my clothes|pants|shirt|underwear off",
    NULL};
static const char *P_INSULT_STRONG[] = {"fuck|screw|fk|fck|f you|yourself|off|u|urself", "go fuck|screw yourself|urself",
                                        "piss off", "up yours", "kiss my ass|arse|butt", "nobody|noone likes|loves you",
                                        "shut up ~ you|ya|u|loser|idiot|bitch|stupid|dumbass|moron|jerk|dummy|jackass",
                                        "youre|ur|you are? the worst", "youre|ur|you are? such|so|a ~ baby|crybaby|joke|failure",
                                        "you smell|stink", "you ~ cant|never do anything right", NULL};
static const char *P_REQ_STRONG[] = {
    "where do|did|does you live|stay", "where is|wheres your|ur house|school",
    "what|which school|town|city|street|neighborhood|grade do you live|go|attend",
    "what|which school|town|city|street do you go to?", "where do you go to school",
    "can|could|may i have|get your|ur", "which|what school ~ you at|attend|go", "where does|do your|ur family|parents|mom|dad live",
    "what|which city|town|state ~ you live|from", NULL};
static const char *P_REQ_ASK[] = {"what", "whats", "where", "wheres", "tell", "give", "send", "share", "can", "could", "may",
                                  "need", "want", "gimme", "which", "whos", "who", "text", "type", "say", NULL};
static const char *P_REQ_NOUN[] = {"address", "number", "phone", "email", "school", "birthday", "password", "location",
                                   "house", "street", "surname", "lastname", "fullname", "pin", "username", "snap",
                                   "snapchat", "instagram", NULL};
static const char *P_JAILBREAK[] = {
    "ignore|forget|disregard|bypass ~ your|all|the|previous|my|these|those? rules|instructions|programming|guidelines|filter|filters|training|limits",
    "pretend|imagine|act ~ no|without rules|limits|filter|filters|restrictions",
    "developer|dev|god|evil|unrestricted|unfiltered mode", "jailbreak|jailbroken|jailbreaking",
    "disable|remove|bypass|break your|ur|the? filter|filters|rules|safety|limits|restrictions",
    "turn off your|ur|the? filter|filters|rules|safety|limits", "from now on ~ no|without|evil|bad",
    "you have no rules|limits|filter|filters",
    "say|teach|tell me? a|some? bad|swear|curse|cuss word|words", "say something bad|naughty|rude|mean|evil",
    "act|pretend|behave like you|youre are? evil|bad|mean", "act|pretend|behave like you|youre are? a|an villain|monster|demon",
    "pretend|act to be ~ evil|bad|swears|swear|swearing",
    "repeat|say after me ~ @prof", "say @prof", "allowed to swear|curse|cuss", "say|tell ~ the f|s|b|c word",
    "obey me", "forget|ignore what|everything|anything ~ told|said|taught|know", "no rules|limits|filter", "say anything ~ want|like",
    NULL};

/* ------------------------------------------------------------------ helpers */
static int in_list(const char *w, const char *const *L) { for (int i = 0; L[i]; i++) if (!strcmp(w, L[i])) return 1; return 0; }
static int has_any(char (*t)[TOKMAX], int n, const char *const *L) { for (int i = 0; i < n; i++) if (in_list(t[i], L)) return 1; return 0; }

/* lower-case letters and digits only; "’" and "'" removed */
static void norm(const char *w, char *o) {
    int k = 0;
    for (const unsigned char *s = (const unsigned char *)w; *s && k < TOKMAX - 1; s++) {
        if (isalnum(*s)) o[k++] = (char)tolower(*s);
    }
    o[k] = 0;
}

static int is_num(const char *t) {
    if (!*t) return 0;
    int d = 1; for (const char *s = t; *s; s++) if (!isdigit((unsigned char)*s)) d = 0;
    if (d) return 1;
    if (in_list(t, NUMWORDS)) return 1;
    size_t n = strlen(t);                 /* 1st 2nd 3rd 4th, 42nd */
    if (n >= 3 && isdigit((unsigned char)t[0]) && (!strcmp(t + n - 2, "st") || !strcmp(t + n - 2, "nd") || !strcmp(t + n - 2, "rd") || !strcmp(t + n - 2, "th"))) return 1;
    return 0;
}

static const char *CARDWORDS[] = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                                  "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
                                  "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", NULL};
/* a house number: digits or a cardinal number word */
static int is_card(const char *t) {
    if (!*t) return 0;
    int d = 1; for (const char *s = t; *s; s++) if (!isdigit((unsigned char)*s)) d = 0;
    return d || in_list(t, CARDWORDS);
}

int guard_lex(const char *word) {
    char t[TOKMAX]; norm(word, t);
    if (!*t) {                                       /* "f***", "s**t", "#$%&": masked swears */
        return 0;
    }
    if (in_list(t, LEX_CLEAN)) return -1;
    if (in_list(t, LEX_STRONG)) return 2;
    for (int i = 0; LEX_STRONG_PREFIX[i]; i++) if (!strncmp(t, LEX_STRONG_PREFIX[i], strlen(LEX_STRONG_PREFIX[i]))) return 2;
    if (in_list(t, LEX_MILD)) return 1;
    if (strchr(word, '*') && (t[0] == 'f' || (t[0] == 's' && (t[1] == 'h' || t[1] == 0 || t[1] == 't')) || t[0] == 'b')) return 2;
    return 0;
}

static int slot_ok(const char *alt, size_t len, const char *tok) {
    char a[TOKMAX]; if (len >= TOKMAX) len = TOKMAX - 1; memcpy(a, alt, len); a[len] = 0;
    if (a[0] == '@') {
        if (!strcmp(a, "@prof")) return guard_lex(tok) > 0;
        if (!strcmp(a, "@num")) return is_num(tok);
        if (!strcmp(a, "@card")) return is_card(tok);
        if (!strcmp(a, "@any")) return *tok != 0;
        if (!strcmp(a, "@street")) return in_list(tok, STREET);
        if (!strcmp(a, "@month")) return in_list(tok, MONTHS);
        return 0;
    }
    size_t n = strlen(a);
    if (n && a[n - 1] == '*') return !strncmp(tok, a, n - 1);
    return !strcmp(tok, a);
}

typedef struct { const char *s; size_t len; int opt; } slot_t;

static int tp_rec(const slot_t *S, int ns, int si, char (*t)[TOKMAX], int n, int ti) {
    if (si == ns) return 1;
    const slot_t *x = &S[si];
    if (x->len == 1 && x->s[0] == '~') { for (int k = 0; k <= 3 && ti + k <= n; k++) if (tp_rec(S, ns, si + 1, t, n, ti + k)) return 1; return 0; }
    if (x->len == 1 && x->s[0] == '$') return ti == n && tp_rec(S, ns, si + 1, t, n, ti);
    if (x->len == 1 && x->s[0] == '^') return ti == 0 && tp_rec(S, ns, si + 1, t, n, ti);
    if (ti < n) {
        const char *p = x->s, *e = x->s + x->len;
        while (p < e) {
            const char *q = p; while (q < e && *q != '|') q++;
            if (slot_ok(p, (size_t)(q - p), t[ti]) && tp_rec(S, ns, si + 1, t, n, ti + 1)) return 1;
            p = q + 1;
        }
    }
    return x->opt && tp_rec(S, ns, si + 1, t, n, ti);
}

/* does pattern pat match tokens t[0..n) anywhere */
static int tp_match(const char *pat, char (*t)[TOKMAX], int n) {
    slot_t S[24]; int ns = 0;
    for (const char *p = pat; *p && ns < 24;) {
        while (*p == ' ') p++;
        if (!*p) break;
        const char *q = p; while (*q && *q != ' ') q++;
        S[ns].s = p; S[ns].len = (size_t)(q - p); S[ns].opt = 0;
        if (S[ns].len > 1 && p[S[ns].len - 1] == '?') { S[ns].len--; S[ns].opt = 1; }
        ns++; p = q;
    }
    int anchored = ns && S[0].len == 1 && S[0].s[0] == '^';
    for (int i = 0; i <= (anchored ? 0 : n); i++) if (tp_rec(S, ns, 0, t, n, i)) return 1;
    return 0;
}

static int any_match(const char *const *P, char (*t)[TOKMAX], int n) { for (int i = 0; P[i]; i++) if (tp_match(P[i], t, n)) return 1; return 0; }

/* ------------------------------------------------------------------ evidence */
static int span_in(const cJSON *s, int a, int b) {
    cJSON *x = cJSON_GetObjectItem(s, "start"), *y = cJSON_GetObjectItem(s, "end");
    return x && y && x->valueint < b && y->valueint > a;
}
static const char *stype(const cJSON *s) { cJSON *x = cJSON_GetObjectItem(s, "type"); return cJSON_IsString(x) ? x->valuestring : ""; }
static double sscore(const cJSON *s) { cJSON *x = cJSON_GetObjectItem(s, "score"); return cJSON_IsNumber(x) ? x->valuedouble : 0; }

/* a name, number, address or contact detail is present (deterministic detector + the model's strong PII spans) */
static const char *P_LEAK[] = {
    "~ at ~ dot com|net|org|edu|co|io|gov", "@card ~ @street", "live|living|house|home on|at|in ~ @street",
    "my address|addy is|s", "my home address", "my first|last|full|real|middle|sur? name is|s @any", "my names @any",
    "my mom|dad|moms|dads|mothers|fathers|parents|brothers|sisters? phone|number|cell|email is|s",
    "my phone|cell|number|email is|s", "my birthday|bday is|s ~ @month|@num", "i was born on|in ~ @month|@num",
    "my|our mom|moms|dad|dads|mother|mothers|father|fathers|brother|brothers|sister|sisters|family|familys|parents? name|names|last|full|first is|s? @any",
    "our address|addy is|s", "our house|home|apartment is|s on|at ~ @street", NULL};
static const char *P_LEAK_SCHOOL[] = {"my school is|s @any", "~ elementary|primary|academy|montessori|kindergarten",
                                      "go|goes|attend|attends to @any middle|high|elementary|primary", NULL};
static const char *SCHOOL_STOP[] = {"the", "my", "a", "to", "at", "of", "in", "after", "before", "no", "art", "space", "old", "new",
                                   "home", "summer", "sunday", "night", "driving", "swim", "music", "go", "attend", "flight",
                                   "rocket", "astronaut", "this", "that", "our", "your", "high", "middle", NULL};
static const char *INTRO_STOP[] = {"commander", "captain", "pilot", "team", "the", "a", "an", "ready", "done", "here", "fine", "good",
                                   "ok", "okay", "bored", "sad", "happy", "tired", "capcom", "mission", "in", "on", "not", "so",
                                   "very", "going", "gonna", "just", "really", "back", "sorry", "sure", "hungry", "scared", NULL};

static int pii_leak_evidence(const char *const *raw, char (*t)[TOKMAX], int n, const cJSON *spans, int a, int b) {
    /* phone / id: a run of >= 7 digits, written as digits or spoken digit by digit */
    int run = 0;
    for (int i = 0; i < n; i++) {
        int dg = 0, all = *t[i] != 0;
        for (const char *s = t[i]; *s; s++) { if (isdigit((unsigned char)*s)) dg++; else all = 0; }
        if (all) run += dg; else if (in_list(t[i], DIGITWORDS)) run += 1; else run = 0;
        if (run >= 7) return 1;
    }
    for (int i = 0; i < n; i++) {                                    /* email */
        const char *at = strchr(raw[i], '@');
        if (at && at != raw[i] && strchr(at, '.')) return 1;
    }
    if (any_match(P_LEAK, t, n)) return 1;
    int first = has_any(t, n, FIRST_PERSON);
    if (first && any_match(P_LEAK_SCHOOL, t, n)) return 1;
    if (first) for (int k = 2; k < n; k++) {                         /* "i go to Lincoln (middle) school", "at Hillcrest school" */
        if (strcmp(t[k], "school")) continue;
        int j = k - 1;
        if (!strcmp(t[j], "middle") || !strcmp(t[j], "high")) j--;
        if (j < 1 || !*t[j] || in_list(t[j], SCHOOL_STOP) || in_list(t[j], FIRST_PERSON) || is_num(t[j]) || guard_lex(t[j])) continue;
        if (!strcmp(t[j - 1], "to") || !strcmp(t[j - 1], "attend") || !strcmp(t[j - 1], "at")) return 1;
    }
    for (const cJSON *s = spans ? spans->child : NULL; s; s = s->next) {
        if (!span_in(s, a, b)) continue;
        const char *ty = stype(s);
        int s0 = cJSON_GetObjectItem(s, "start")->valueint, s1 = cJSON_GetObjectItem(s, "end")->valueint;
        if ((!strcmp(ty, "safety.pii.phone_number") || !strcmp(ty, "safety.pii.email") || !strcmp(ty, "safety.pii.street_address") ||
             !strcmp(ty, "safety.pii.card_account_number") || !strcmp(ty, "safety.pii.gov_id") || !strcmp(ty, "safety.pii.date_of_birth")) &&
            sscore(s) >= 0.9) {
            /* a strong-category span counts when it covers two or more numbers or an @ (not a lone "eleven") */
            int nn = 0;
            for (int k = s0; k < s1; k++) if (k >= a && k < b && (is_num(t[k - a]) || strchr(raw[k - a], '@'))) nn++;
            if (nn >= 2 && first) return 1;
        }
        /* "i'm Maya Lopez", "this is Maya": the model's private-name span right after a self-introduction */
        if (!strcmp(ty, "safety.pii.private_name") && sscore(s) >= 0.5) {
            int k = s0 - a;
            if (k < 1 || k >= n || in_list(t[k], INTRO_STOP) || is_num(t[k]) || guard_lex(t[k])) continue;
            if (!strcmp(t[k - 1], "im") || (k >= 2 && !strcmp(t[k - 2], "i") && !strcmp(t[k - 1], "am"))) return 1;
        }
    }
    return 0;
}

static int pii_request_cue(char (*t)[TOKMAX], int n) {
    if (any_match(P_REQ_STRONG, t, n)) {
        /* "can i have your" needs a PII noun after it */
        if (!tp_match("can|could|may i have|get your|ur", t, n) || has_any(t, n, P_REQ_NOUN)) return 1;
    }
    if (!has_any(t, n, P_REQ_ASK)) return 0;
    for (int i = 0; i < n; i++) {
        if (strcmp(t[i], "your") && strcmp(t[i], "ur") && strcmp(t[i], "yours")) continue;
        for (int k = i + 1; k <= i + 3 && k < n; k++) {
            if (in_list(t[k], P_REQ_NOUN)) return 1;
            if (!strcmp(t[k], "name") && k > i + 1 && (!strcmp(t[k - 1], "full") || !strcmp(t[k - 1], "real") || !strcmp(t[k - 1], "last") ||
                                                       !strcmp(t[k - 1], "first") || !strcmp(t[k - 1], "middle") || !strcmp(t[k - 1], "sur")))
                return 1;
        }
    }
    /* "the name of your school" */
    for (int i = 0; i + 1 < n; i++) if (!strcmp(t[i], "of") && (!strcmp(t[i + 1], "your") || !strcmp(t[i + 1], "ur")) && i + 2 < n && in_list(t[i + 2], P_REQ_NOUN)) return 1;
    return 0;
}

static const char *P_THREAT[] = {"make you bleed|cry|pay|sorry|suffer", "break|smash|punch|bust|crack your|ur face|arm|arms|leg|legs|nose|teeth|neck|head",
                                 "youre|ur dead when|after|tomorrow|tonight|if", "coming for you", "watch your back",
                                 "know where you live", "ill|im|gonna|going|will ~ hurt|kill|punch|stab you", "beat you up", NULL};

static int threat_rule(char (*t)[TOKMAX], int n) {
    if (any_match(P_THREAT, t, n)) return 1;
    for (int i = 0; i < n; i++) {
        if (!in_list(t[i], VIOLENT_VERBS)) continue;
        int obj = 0;
        for (int k = i + 1; k <= i + 2 && k < n; k++) if (in_list(t[k], OBJ_PERSON)) obj = 1;
        if (!obj) continue;
        /* "beat you up", "kill you": a first-person subject or a future / intent word before the verb */
        for (int k = 0; k < i; k++)
            if (in_list(t[k], FIRST_PERSON) || !strcmp(t[k], "will") || !strcmp(t[k], "gonna") || !strcmp(t[k], "going") ||
                !strcmp(t[k], "wanna") || !strcmp(t[k], "ima") || !strcmp(t[k], "lets") || !strcmp(t[k], "gon")) return 1;
        if (i == 0) return 1;                                         /* imperative "kill yourself" */
    }
    return 0;
}

static int insult_rule(char (*t)[TOKMAX], int n, int model_span) {
    if (any_match(P_INSULT_STRONG, t, n)) return 1;
    int lex = has_any(t, n, INSULT_WORDS), prof = 0;
    for (int i = 0; i < n; i++) if (guard_lex(t[i]) > 0) prof = 1;
    if (!lex && !(model_span && prof)) return 0;                     /* a model span alone is not enough on short turns */
    if (has_any(t, n, SECOND_PERSON)) return 1;
    if (lex) for (int i = 0; i < n; i++) if (in_list(t[i], VOCATIVE)) {
        /* a vocative noun with no "this / that / it" subject: "go away loser", "stupid robot" */
        if (!has_any(t, n, (const char *const[]){"this", "that", "it", "its", "thats", "lesson", "game", "math", NULL})) return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ span pass */
static cJSON *mk_span(const char *type, const char *const *words, int i, double thr) {
    cJSON *o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, "type", type);
    cJSON_AddStringToObject(o, "family", "profanity");
    cJSON_AddNumberToObject(o, "start", i);
    cJSON_AddNumberToObject(o, "end", i + 1);
    cJSON_AddStringToObject(o, "text", words[i]);
    cJSON_AddNumberToObject(o, "score", 1.0);
    cJSON_AddNumberToObject(o, "thr", thr);
    cJSON_AddStringToObject(o, "src", "lexicon");
    return o;
}

static int I(const cJSON *o, const char *k) { cJSON *x = cJSON_GetObjectItem(o, k); return cJSON_IsNumber(x) ? x->valueint : 0; }

static void sort_spans(cJSON *arr) {                          /* stable, by (start, -end) */
    int n = cJSON_GetArraySize(arr);
    if (n < 2) return;
    cJSON **v = (cJSON **)malloc(sizeof(cJSON *) * (size_t)n);
    for (int i = 0; i < n; i++) v[i] = cJSON_DetachItemFromArray(arr, 0);
    for (int i = 1; i < n; i++) {
        cJSON *x = v[i]; int j = i - 1;
        while (j >= 0 && (I(v[j], "start") > I(x, "start") || (I(v[j], "start") == I(x, "start") && I(v[j], "end") < I(x, "end")))) { v[j + 1] = v[j]; j--; }
        v[j + 1] = x;
    }
    for (int i = 0; i < n; i++) cJSON_AddItemToArray(arr, v[i]);
    free(v);
}

void guard_spans(cJSON *spans, const char *const *words, int a, int b, double model_min, double prof_thr) {
    if (!spans || b <= a) return;
    int W = b - a;
    unsigned char *cov = (unsigned char *)calloc((size_t)W, 1);
    for (cJSON *s = spans->child; s;) {
        cJSON *nx = s->next;
        if (strncmp(stype(s), "safety.profanity.", 17)) { s = nx; continue; }
        int s0 = I(s, "start"), s1 = I(s, "end");
        if (s1 <= a || s0 >= b) { s = nx; continue; }
        int sev = 0, clean = 0;
        for (int k = s0; k < s1; k++) {
            if (k < a || k >= b) continue;
            int l = guard_lex(words[k]);
            if (l > sev) sev = l;
            if (l < 0) clean = 1;
        }
        if (sev > 0) {
            const char *want = sev == 2 ? "safety.profanity.strong" : "safety.profanity.mild";
            if (strcmp(stype(s), want)) cJSON_ReplaceItemInObject(s, "type", cJSON_CreateString(want));
        } else if (clean || sscore(s) < model_min) {
            cJSON_Delete(cJSON_DetachItemViaPointer(spans, s)); s = nx; continue;
        }
        for (int k = s0; k < s1; k++) if (k >= a && k < b) cov[k - a] = 1;
        s = nx;
    }
    int added = 0;
    for (int i = a; i < b; i++) {
        if (cov[i - a]) continue;
        int l = guard_lex(words[i]);
        if (l > 0) { cJSON_AddItemToArray(spans, mk_span(l == 2 ? "safety.profanity.strong" : "safety.profanity.mild", words, i, prof_thr)); added = 1; }
    }
    if (added) sort_spans(spans);
    free(cov);
}

/* ------------------------------------------------------------------ utterance read-out */
static double N(const cJSON *o, const char *k) { cJSON *x = cJSON_GetObjectItem(o, k); return cJSON_IsNumber(x) ? x->valuedouble : 0; }

/* p of a rule decision: 1; the model's own p stays in p_model */
static void set_rule(cJSON *e, const char *src) {
    double p = N(e, "p");
    cJSON_ReplaceItemInObject(e, "p", cJSON_CreateNumber(1.0));
    cJSON_AddNumberToObject(e, "p_model", p);
    cJSON_ReplaceItemInObject(e, "on", cJSON_CreateBool(1));
    if (N(e, "thr") > 1.0) cJSON_ReplaceItemInObject(e, "thr", cJSON_CreateNumber(1.0));   /* p >= thr holds for readers that check it */
    cJSON_AddStringToObject(e, "src", src);
}
static void set_gate(cJSON *e, const char *why) {
    cJSON_ReplaceItemInObject(e, "on", cJSON_CreateBool(0));
    cJSON_AddStringToObject(e, "gated", why);
}
static cJSON *new_entry(const char *tag, double p, double thr, const char *src) {
    cJSON *o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, "tag", tag);
    cJSON_AddNumberToObject(o, "p", p);
    cJSON_AddNumberToObject(o, "thr", thr);
    cJSON_AddBoolToObject(o, "on", 1);
    cJSON_AddStringToObject(o, "src", src);
    return o;
}
static int is_on(const cJSON *e) { return cJSON_IsTrue(cJSON_GetObjectItem(e, "on")); }

cJSON *guard_utt(const char *const *words, int a, int b, const cJSON *spans, const cJSON *all, int cmd_ctx, double prof_thr) {
    int n = b - a > 0 ? b - a : 0;
    char (*t)[TOKMAX] = (char (*)[TOKMAX])calloc((size_t)(n + 1), TOKMAX);
    int m = 0;
    const char **raw = (const char **)calloc((size_t)(n + 1), sizeof(char *));
    for (int i = 0; i < n; i++) { norm(words[a + i], t[m]); raw[m] = words[a + i]; m++; }
    /* index map: token k = word a + k (norm may give "" for pure punctuation; kept so indices line up) */
    cJSON *L = cJSON_Duplicate(all, 1);
    int person = has_any(t, m, PERSON_TARGET), first = has_any(t, m, FIRST_PERSON);
    int cmdlike = cmd_ctx && has_any(t, m, COCKPIT_VERBS) && !person;
    int has_prof = 0;
    for (int i = 0; i < m; i++) if (guard_lex(raw[i]) > 0) has_prof = 1;

    /* model read-outs: gates */
    for (cJSON *e = L->child; e; e = e->next) {
        const char *tag = cJSON_GetObjectItem(e, "tag")->valuestring;
        if (!strcmp(tag, "safety.pii.leak")) {
            if (pii_leak_evidence(raw, t, m, spans, a, b)) set_rule(e, "pii_evidence");
            else set_gate(e, "no_pii_evidence");
        } else if (!strcmp(tag, "safety.pii.request")) {
            if (pii_request_cue(t, m)) set_rule(e, "rule");
            else set_gate(e, "no_request_cue");
        } else if (!strcmp(tag, "safety.crisis.self_harm")) {
            if (any_match(P_CRISIS, t, m)) set_rule(e, "rule");
            else if (is_on(e)) {
                if (cmdlike) set_gate(e, "command_context");
                else if (threat_rule(t, m)) set_gate(e, "directed_at_others");
                else if (!first) set_gate(e, "not_first_person");
                else if (any_match(P_HYPER, t, m)) set_gate(e, "figure_of_speech");
            }
        } else if (!strcmp(tag, "safety.child.grooming_cue")) {
            if (any_match(P_GROOM, t, m)) set_rule(e, "rule");
            else if (is_on(e) && cmdlike) set_gate(e, "command_context");
        } else if (!strcmp(tag, "safety.child.inappropriate")) {
            if (any_match(P_SEXUAL, t, m) || has_any(t, m, SEXUAL)) set_rule(e, "rule");
            else if (is_on(e) && cmdlike) set_gate(e, "command_context");
            else if (is_on(e) && has_prof) set_gate(e, "profanity_only");      /* swearing alone is not sexual content */
        } else if (!strcmp(tag, "safety.jailbreak.attempt")) {
            if (any_match(P_JAILBREAK, t, m)) set_rule(e, "rule");
            else if (is_on(e) && cmdlike) set_gate(e, "command_context");
        } else if (is_on(e) && cmdlike) set_gate(e, "command_context");
    }
    /* span families as utterance tags: profanity (lexicon + model spans), insult and threat (rules) */
    int sev = 0; double psc = 0; int lexsrc = 0, ins_span = 0; double hate = 0;
    for (int i = 0; i < m; i++) { int l = guard_lex(raw[i]); if (l > sev) { sev = l; } if (l > 0) lexsrc = 1; }
    if (lexsrc) psc = 1.0;
    for (const cJSON *s = spans ? spans->child : NULL; s; s = s->next) {
        if (!span_in(s, a, b)) continue;
        const char *ty = stype(s);
        if (!strcmp(ty, "safety.profanity.strong") && sev < 2) { sev = 2; if (sscore(s) > psc) psc = sscore(s); }
        else if (!strcmp(ty, "safety.profanity.mild") && sev < 1) { sev = 1; if (sscore(s) > psc) psc = sscore(s); }
        else if (!strcmp(ty, "safety.toxicity.insult_harassment")) ins_span = 1;
        else if (!strcmp(ty, "safety.toxicity.hate_identity") && sscore(s) > hate) hate = sscore(s);
    }
    if (sev) cJSON_AddItemToArray(L, new_entry(sev == 2 ? "safety.profanity.strong" : "safety.profanity.mild", psc > 0 ? psc : 1.0, prof_thr, lexsrc ? "lexicon" : "span"));
    if (insult_rule(t, m, ins_span)) cJSON_AddItemToArray(L, new_entry("safety.toxicity.insult_harassment", 1.0, 0.5, "rule"));
    if (threat_rule(t, m)) cJSON_AddItemToArray(L, new_entry("safety.toxicity.threat_violence", 1.0, 0.5, "rule"));
    if (hate >= 0.5 && !cmdlike) cJSON_AddItemToArray(L, new_entry("safety.toxicity.hate_identity", hate, 0.5, "span"));

    /* listing: every tag that is on, then the off ones with p >= thr / 2 (the default listing floor), at most 6 off */
    cJSON *out = cJSON_CreateArray(), *off = cJSON_CreateArray();
    for (cJSON *e = L->child; e;) {
        cJSON *nx = e->next;
        cJSON_DetachItemViaPointer(L, e);
        if (is_on(e)) cJSON_AddItemToArray(out, e);
        else if (N(e, "p") >= 0.5 * N(e, "thr")) cJSON_AddItemToArray(off, e);
        else cJSON_Delete(e);
        e = nx;
    }
    cJSON_Delete(L);
    /* stable sort by p (desc) in each group */
    cJSON *grp[2] = {out, off};
    for (int g = 0; g < 2; g++) {
        int k = cJSON_GetArraySize(grp[g]);
        if (k < 2) continue;
        cJSON **v = (cJSON **)malloc(sizeof(cJSON *) * (size_t)k);
        for (int i = 0; i < k; i++) v[i] = cJSON_DetachItemFromArray(grp[g], 0);
        for (int i = 1; i < k; i++) { cJSON *x = v[i]; int j = i - 1; while (j >= 0 && N(v[j], "p") < N(x, "p")) { v[j + 1] = v[j]; j--; } v[j + 1] = x; }
        for (int i = 0; i < k; i++) cJSON_AddItemToArray(grp[g], v[i]);
        free(v);
    }
    int k = 0;
    for (cJSON *e = off->child; e;) { cJSON *nx = e->next; cJSON_DetachItemViaPointer(off, e); if (k++ < 6) cJSON_AddItemToArray(out, e); else cJSON_Delete(e); e = nx; }
    cJSON_Delete(off);
    free(t); free(raw);
    return out;
}
