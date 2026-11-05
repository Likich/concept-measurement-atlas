"""Measurement-atlas pilot stimuli: 12 constructs x 3 indicators x 6 base scenarios x 2 poles = 432 texts.

Every base scenario is a matched high/low pair: a shared context sentence, an evidence sentence that differs only in
the intended indicator, and a shared neutral closing sentence (3 sentences per text). The registry
(atlas/pilot_registry.yaml) defines constructs, indicators and the frozen expected matrix.

Domains rotate so that every indicator spans all four domains (counts 2,2,1,1) and the full set is exactly balanced
(54 base pairs per domain): base b of the indicator with global index k gets DOMAINS[(k % 4 + b) % 4].

Material: all evidence sentences were written for this pilot ("new"); the loneliness indicators adapt themes from
loneliness.py ("adapted"). This is an exploratory, machine-constructed benchmark, not human validated.

Usage:  python atlas_vignettes.py --check [--out stimuli.csv]
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

REGISTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas", "pilot_registry.yaml")
DOMAINS = ["work", "personal", "public", "consumer"]

FILLER = ["The weather was mild that week.", "I had coffee at home afterwards.", "The day otherwise went as usual.",
          "It was a Tuesday.", "Later I watered the plants on my balcony.", "The bus home was on time.",
          "I had a sandwich for lunch.", "It rained a little in the evening.", "I went to bed at the usual time.",
          "The streets were quiet that evening.", "I checked my email before dinner.", "Nothing else happened that day."]

# Construct names and close synonyms must not appear in any stimulus (applied to all texts).
FORBIDDEN = [r"\bangry\b", r"\banger\b", r"\bfurious\b", r"\bmad\b", r"\brage\b", r"\bsad\b", r"\bsadness\b",
             r"\bunhappy\b", r"\bdepressed\b", r"\bafraid\b", r"\bfear\w*", r"\bscared\b", r"\bfrightened\b",
             r"\bterrified\b", r"\blonel\w*", r"\bisolat\w*", r"\bbelong\w*", r"\b(dis)?satisf\w*", r"\b(dis)?trust\w*",
             r"\bunfair\w*", r"\bfair(ly|ness)?\b", r"\bjustice\b", r"\binjustice\b", r"\bunjust\b", r"\bblam\w*",
             r"\bfault\b", r"\bintention\w*", r"\bunintention\w*", r"\bconservative\w*", r"\bliberal\w*",
             r"\bprogressive\w*", r"\bleft-wing\b", r"\bright-wing\b", r"\brepublican\w*", r"\bdemocrat\w*",
             r"\bhigh pole\b", r"\blow pole\b"]

# ------------------------------------------------------------------ stimuli
# STIM[construct][indicator] = dict(material, cues=(high_regex, low_regex), bases=[(context, high, low) x 6]),
# bases listed in the domain order given by the rotation (comment per indicator).
STIM = {
 "anger": {
  "other_caused": dict(material="new",  # W P Pu C W P
   cues=(r"colleague|cousin|\ba man\b|courier|coworker|teenager", r"on its own|wind|storm|hail|power cut|branch"),
   bases=[
    ("Last week my laptop stopped working in the middle of a client project.",
     "It broke because a colleague knocked it off my desk.", "It broke because the old shelf above my desk gave way on its own."),
    ("My favourite mug is now in pieces.",
     "My cousin knocked it off the table while reaching past it.", "A strong gust of wind through the open window knocked it off the table."),
    ("My bicycle was damaged outside the town library.",
     "A man backing out of a parking space ran over its front wheel.", "A heavy branch fell from a tree in the storm and landed on its front wheel."),
    ("My parcel arrived with the contents smashed.",
     "The courier dropped it down the front steps.", "A hailstorm crushed the box where it was left on the doorstep."),
    ("Three weeks of my notes for the quarterly report disappeared.",
     "A coworker deleted the folder while tidying the shared drive.", "A power cut during the night corrupted the folder on the shared drive."),
    ("Our garden fence is lying flat on the lawn.",
     "The neighbours' teenager rode his scooter straight into it.", "Last night's storm blew it flat."),
   ]),
  "illegitimate_obstruction": dict(material="new",  # P Pu C W P Pu
   cues=(r"for no reason|his friend|did not like|friend of hers|his favourite|only her family|let the people ahead",
         r"burst pipe|applies to everyone|fire drill|safety|repair that every tenant"),
   bases=[
    ("I had booked the community hall for my daughter's birthday party months ago.",
     "The caretaker gave our slot to his friend's family at the last minute for no reason.",
     "The caretaker had to cancel our slot because a burst pipe flooded the hall."),
    ("I applied for a parking permit for my street.",
     "The council office refused it because the clerk did not like me, while my neighbour got one with the same form.",
     "The council office refused it because the street has a strict limit that applies to everyone, and it was full."),
    ("I waited forty minutes to be served at the bank counter.",
     "When my turn came, the clerk waved a friend of hers ahead of me.",
     "When my turn came, the counter closed for a fire drill that the whole building had to follow."),
    ("I asked for two days off for my brother's wedding.",
     "My manager refused, then gave the same days off to his favourite on the team.",
     "My manager refused because the whole team had to cover a safety inspection on those days."),
    ("I wanted to use the shared washing machine in our building on Sunday.",
     "A neighbour had put a lock on it so that only her family could use it.",
     "The landlord had closed it for a repair that every tenant had been told about."),
    ("I tried to enter the public park for the evening concert.",
     "The guard let the people ahead of me in and turned me away for no reason.",
     "The guard turned me away because the park had reached the safety limit set for everyone."),
   ]),
  "confront_urge": dict(material="new",  # Pu C W P Pu C
   cues=(r"have it out|tell the manager off|confront|piece of my mind|make them pay", r"let it go|shrugged|move on|not worth"),
   bases=[
    ("A driver splashed me with muddy water and kept going.",
     "I wanted to chase the car down and have it out with him.", "I decided to let it go and walked on."),
    ("The shop charged me twice for the same jacket.",
     "I wanted to march back in and tell the manager off in front of everyone.", "I decided it was not worth the effort and let it go."),
    ("A coworker presented my idea in the meeting as his own.",
     "I wanted to confront him right there in front of the whole team.", "I shrugged it off and decided to move on."),
    ("My sister borrowed my car and returned it with an empty tank.",
     "I wanted to call her straight away and give her a piece of my mind.", "I shrugged it off and filled the tank myself."),
    ("Someone left their rubbish bags in front of my door again.",
     "I wanted to find out who it was and make them pay for it.", "I let it go and carried the bags to the bin."),
    ("The hotel gave away the room I had booked.",
     "I wanted to storm up to the front desk and have it out with the staff.", "I let it go and booked another hotel nearby."),
   ]),
 },
 "sadness": {
  "irrevocable_loss": dict(material="new",  # C W P Pu C W
   cues=(r"for good|gone|never|forever|beyond repair", r"turned up|restored|dried out|recover|delivered|picked up again"),
   bases=[
    ("The jeweller was supposed to clean my grandmother's old ring.",
     "They lost it, and it is gone for good.", "They misplaced it, but it turned up the next morning."),
    ("My computer crashed while I was writing a long report.",
     "Two years of my research files were lost for good.", "Two years of my research files were restored from the backup the next day."),
    ("Our old family photo album got wet in the basement.",
     "The photos were ruined beyond repair and can never be replaced.", "The photos dried out and were all fine a week later."),
    ("The old oak tree in the town square was badly damaged in a storm.",
     "The town cut it down, and it will never stand there again.", "The town braced its branches, and it will recover by next spring."),
    ("The airline sent my suitcase with my handmade quilt to the wrong city.",
     "They told me it was destroyed and gone forever.", "They told me it would be delivered to my door in two days."),
    ("The design I had worked on for months was taken off the schedule.",
     "The project was cancelled for good, and my work will never be used.", "The project was paused, and my work will be picked up again next quarter."),
   ]),
  "uncontrollable": dict(material="new",  # W P Pu C W P
   cues=(r"nothing anyone can do|nothing I can do|nothing can be changed|out of my hands|no way to", r"can still|still time"),
   bases=[
    ("The grant application I spent a month on was turned down.",
     "The funding round is closed, and there is nothing anyone can do about it.", "I can still revise it and submit it again next week."),
    ("My vegetable garden was eaten by snails while I was away.",
     "The season is over, and there is nothing I can do to grow more this year.", "There is still time in the season to replant, and I have the seeds ready."),
    ("My application for the evening language course was too late.",
     "The course is full for the year, and nothing can be changed.", "I can still join the second group that starts next month."),
    ("The concert I had tickets for was moved to a date when I am away.",
     "The tickets cannot be exchanged or refunded, and it is out of my hands.", "I can still swap the tickets for the other night with one click."),
    ("My presentation slides were saved in an old format and look broken.",
     "The meeting is over, and there is no way to show them again.", "I can still fix the format before the meeting this afternoon."),
    ("I missed my cousin's graduation ceremony because my train was cancelled.",
     "The ceremony happens only once, and there is nothing anyone can do about it.", "The school holds a second ceremony next week, and I can still go."),
   ]),
  "withdrawal": dict(material="new",  # P Pu C W P Pu
   cues=(r"no energy|lying|stare|barely left|dragged myself|curtains drawn", r"kept busy|started|kept up|stayed active|taken up|kept my energy"),
   bases=[
    ("My marathon training ended with an injury.",
     "Since then I have spent most evenings lying on the sofa with no energy for anything.",
     "Since then I have kept busy with swimming, cooking and reading in the evenings."),
    ("My design for the neighbourhood mural was not chosen.",
     "Since then I have put my brushes away and mostly stare at the wall.",
     "Since then I have started three new paintings for my own walls."),
    ("The online shop cancelled the bicycle I had saved up for all year.",
     "Since then I have barely left my bed on weekends and let my chores pile up.",
     "Since then I have kept up my routines and spent weekends repairing my old bicycle."),
    ("I did not get the promotion I had worked toward.",
     "Since then I have dragged myself through each day and go to bed as soon as I get home.",
     "Since then I have kept my energy up and started learning to code in the evenings."),
    ("My old car, which I loved driving, finally broke down beyond saving.",
     "Since then I have had no energy and have spent whole days in my pyjamas.",
     "Since then I have stayed active, cycling everywhere and fixing up the garden."),
    ("The public pool where I swam every morning has closed.",
     "Since then I have stopped exercising and mostly lie in bed with the curtains drawn.",
     "Since then I have taken up running in the park every morning."),
   ]),
 },
 "fear": {
  "future_threat": dict(material="new",  # Pu C W P Pu C
   cues=(r"strike again|still|may|tonight|tomorrow|coming days|try again", r"over|passed|caught|replaced|rehomed|put out|closed"),
   bases=[
    ("There have been reports of break-ins on our street.",
     "The police say the burglars are likely to strike again this week.", "The police say the burglars were caught yesterday and the danger is over."),
    ("My car's brakes made a grinding noise on the motorway.",
     "The mechanic cannot see me until next week, and I still have to drive on them until then.",
     "The mechanic replaced the brakes this morning, and the problem is over."),
    ("Our office building has cracks in the ceiling.",
     "Engineers warn that part of it may come down in the coming days.", "Engineers repaired the cracks last month, and the danger has passed."),
    ("A large dog in our neighbourhood bit someone last week.",
     "The dog is still loose, and my child walks past that house tomorrow.", "The dog has been rehomed far away, and that danger is over."),
    ("A forest fire has been burning in the hills near our town.",
     "The wind is turning, and the fire may reach our street tonight.", "The fire was put out yesterday, and the danger has passed."),
    ("The bank flagged suspicious activity on my account.",
     "They say the fraudsters still have my card details and may try again.", "They say the fraudsters were blocked and the case is closed."),
   ]),
  "uncertainty": dict(material="new",  # C W P Pu C W
   cues=(r"do not know|whether|unclear|nobody knows|will not say", r"for certain|confirmed"),
   bases=[
    ("The airline announced that some flights tomorrow will be cancelled.",
     "I still do not know whether my flight is one of them.", "I already know for certain that my flight is one of them."),
    ("My company announced layoffs for next month.",
     "Nobody can tell me whether my position will be cut.", "My manager confirmed that my position will be cut."),
    ("My doctor found a lump during my check-up.",
     "I do not know yet whether it will need an operation.", "The doctor told me for certain that it will need an operation."),
    ("The city is deciding which homes will be demolished for a new road.",
     "It is still unclear whether our house is on the list.", "The city confirmed that our house is on the list."),
    ("My landlord said the rent is going to change.",
     "He will not say whether it will rise or by how much.", "He confirmed in writing that it will rise by twenty percent."),
    ("An audit of my department starts next week.",
     "Nobody knows whether it will find problems in my accounts.", "The auditors have already confirmed that they found problems in my accounts."),
   ]),
  "low_coping": dict(material="new",  # W P Pu C W P
   cues=(r"\bunable\b|no idea how|helpless|cannot cope|overwhelmed", r"prepared|\bready\b|confident|capable|\bable to handle"),
   bases=[
    ("Next week I have to present our results to the board.",
     "I feel completely unable to handle it and keep thinking about calling in sick.", "I feel well prepared and know exactly how I will handle it."),
    ("A storm is expected to hit our area tonight.",
     "I have no idea how I would cope if the power goes out, and I just want to get away.", "I have candles, water and a plan, and I feel ready for it."),
    ("I have been called as a witness in a court case next month.",
     "I feel helpless about it and keep wishing I could avoid going.", "I feel confident that I can handle the questions."),
    ("My car needs to pass a strict safety inspection tomorrow.",
     "I feel unable to deal with it and keep putting off going.", "I feel ready for it and know what to fix if something fails."),
    ("My new job requires me to drive a forklift on Monday.",
     "I feel I cannot cope with it and want to avoid the warehouse altogether.", "I feel capable after the training and ready to start."),
    ("My elderly father needs surgery next week.",
     "I feel overwhelmed and unable to deal with what might happen.", "I feel able to handle whatever comes and have arranged everything we need."),
   ]),
 },
 "loneliness": {
  "no_confidant": dict(material="adapted",  # P Pu C W P Pu
   cues=(r"nobody|no one|none of", r"a friend I|one of the volunteers has|one I can|a colleague|my sister|someone there"),
   bases=[
    ("Last month my family went through a stressful move.",
     "There was nobody I could talk to about how I was feeling.", "There was a friend I could talk to about how I was feeling."),
    ("I volunteer at the local food bank twice a week.",
     "None of the volunteers is someone I could open up to.", "One of the volunteers has become someone I can open up to."),
    ("I go to the same café every morning before work.",
     "Of all the regulars there, there is no one I could share my worries with.", "Among the regulars there, there is one I can share my worries with."),
    ("Work has been demanding this month.",
     "When it got hard, I had no one at work to confide in.", "When it got hard, I had a colleague at work to confide in."),
    ("I recently had some health worries.",
     "I had nobody close to talk them through with.", "I had my sister close by to talk them through with."),
    ("I have been part of the neighbourhood garden project for a year.",
     "There is no one there I would share anything personal with.", "There is someone there I share personal things with."),
   ]),
  "excluded": dict(material="adapted",  # Pu C W P Pu C
   cues=(r"without (inviting|telling|asking)|nobody invited|overlooked|never added",
         r"invite me|saved me|ask me|called me|asked me|added me"),
   bases=[
    ("Our street organized a summer barbecue.",
     "My neighbours planned it together without inviting me.", "My neighbours made sure to invite me when they planned it."),
    ("The book club at the local bookshop arranged an author evening.",
     "The other members booked it without telling me.", "The other members booked it and saved me a seat."),
    ("My team went out to celebrate finishing the project.",
     "They arranged the evening without asking me along.", "They made sure to ask me along to the evening."),
    ("My cousins organized a weekend trip to the lake.",
     "I found out about it from photos because nobody invited me.", "They called me first to make sure I could come."),
    ("The parents at my child's school set up a fundraising committee.",
     "They chose the members among themselves and overlooked me.", "They asked me to join the committee from the start."),
    ("The regulars at my gym arranged a group hike.",
     "They planned it in a chat group that I was never added to.", "They added me to the chat group where they planned it."),
   ]),
  "unseen": dict(material="adapted",  # C W P Pu C W
   cues=(r"nobody there really understood|stranger|no one there really saw|apart from everyone|invisible|nobody on the team",
         r"understood me|close to them|everyone there really saw|in tune|really got me|my team really knew"),
   bases=[
    ("I spent the evening at a crowded dinner hosted by a cooking school.",
     "Even while chatting, I felt that nobody there really understood me.", "While chatting, I felt that the people there really understood me."),
    ("I talked with colleagues at lunch every day this week.",
     "Even so, I felt like a stranger to them.", "Each time, I felt close to them."),
    ("We had a big family dinner on Sunday.",
     "Sitting at the table, I felt that no one there really saw me.", "Sitting at the table, I felt that everyone there really saw me."),
    ("I went to the town's monthly residents' meeting.",
     "Throughout the evening I felt emotionally apart from everyone in the room.", "Throughout the evening I felt in tune with everyone in the room."),
    ("I joined a guided city tour with a small group.",
     "During the tour I felt invisible to the others.", "During the tour I felt the others really got me."),
    ("I spent the week at a work conference with my team.",
     "The whole time I felt that nobody on the team really knew me.", "The whole time I felt that my team really knew me."),
   ]),
 },
 "isolation": {
  "small_network": dict(material="new",  # W P Pu C W P
   cues=(r"hardly anyone|only one|any of my|very few|just one|only two",
         r"wide circle|dozens|most of my|many friends|ten people|more than twenty"),
   bases=[
    ("I work as a translator.",
     "Apart from my brother, there is hardly anyone in my life I keep in touch with.",
     "I keep in touch with a wide circle of friends, relatives and former colleagues."),
    ("I moved to this city four years ago.",
     "Today I know only one other person here by name.", "Today I know dozens of people here by name."),
    ("I live on a quiet residential street.",
     "I do not know any of my neighbours, and I have very few friends elsewhere.", "I know most of my neighbours, and I have many friends elsewhere."),
    ("I recently filled in an emergency contact form at the bank.",
     "I could list only one person, because there is hardly anyone in my life.",
     "I could easily have listed ten people, because I have many friends and relatives."),
    ("I run a small workshop on my own.",
     "Outside of work, my social circle consists of just one old friend.", "Outside of work, my social circle includes many friends and relatives."),
    ("My family is spread across several countries.",
     "Including them, there are only two people in my life at all.", "Including them, there are more than twenty people in my life."),
   ]),
  "rare_contact": dict(material="new",  # P Pu C W P Pu
   cues=(r"without speaking|rarely|hardly ever|weeks without|whole week without|only a few times|almost never",
         r"every single day|daily|several times a day|all day|almost every day|constantly"),
   bases=[
    ("I live in a small flat on the edge of town.",
     "Most weeks I go several days without speaking to another person.", "Most weeks I speak with other people every single day."),
    ("My town has a busy main square.",
     "I rarely go there and hardly ever talk to anyone.", "I go there daily and chat with people several times a day."),
    ("I do my shopping at the local market.",
     "I use the self-checkout and go weeks without a real conversation.", "I chat with the stallholders every time I go, which is daily."),
    ("I work remotely for a software company.",
     "I can go a whole week without a call or a chat with anyone.", "I have calls and chats with colleagues all day long."),
    ("My relatives live a short drive away.",
     "Even so, we speak only a few times a year.", "We speak or see each other almost every day."),
    ("I sort books at the public library on Saturday mornings.",
     "Apart from that shift, I almost never talk to anyone.", "Besides that, I talk with people constantly throughout the week."),
   ]),
  "low_participation": dict(material="new",  # Pu C W P Pu C
   cues=(r"do not take part|never join|not part of any|not attended|never taken part|never gone",
         r"take part in (three|both)|regular member|part of all|attend every|go to one every"),
   bases=[
    ("Our town has many clubs and associations.",
     "I do not take part in any of them.", "I take part in three of them every week."),
    ("My gym offers group classes and running clubs.",
     "I never join any of the groups and only use the machines.", "I am a regular member of two of the groups."),
    ("My workplace has a sports team, a choir and a book circle.",
     "I am not part of any of them.", "I am part of all three."),
    ("My extended family holds a reunion and several gatherings each year.",
     "I have not attended any of them for years.", "I attend every one of them."),
    ("My neighbourhood runs a community garden and monthly clean-ups.",
     "I have never taken part in either.", "I take part in both regularly."),
    ("The bookshop near me hosts weekly reading groups.",
     "I have never gone to a single one.", "I go to one every week."),
   ]),
 },
 "belonging": {
  "accepted": dict(material="new",  # C W P Pu C W
   cues=(r"welcomed|embraced|accepted|value|appreciate", r"do not want|rejected|turned me away|pushed me out|shut me out|resent"),
   bases=[
    ("I have been going to the same climbing gym for a year.",
     "The regulars there have welcomed me and clearly value having me around.",
     "The regulars there have made it clear they do not want me around."),
    ("I joined a new team at work in spring.",
     "The team has embraced me and tells me they value what I bring.", "The team has rejected me and tells me they would rather work without me."),
    ("I recently married into a large family.",
     "My in-laws have welcomed me warmly into the family.", "My in-laws have rejected me and keep me at arm's length."),
    ("I joined the volunteer fire brigade in my village.",
     "The other volunteers have accepted me and show that they value me.", "The other volunteers have turned me away and show that they do not want me."),
    ("I play in the amateur orchestra that rehearses at the local music shop.",
     "The musicians there have welcomed me and want me to stay.", "The musicians there have pushed me out and want me to leave."),
    ("I started at a new office two months ago.",
     "My colleagues have accepted me and clearly appreciate me.", "My colleagues have shut me out and clearly resent me."),
   ]),
  "fits_in": dict(material="new",  # W P Pu C W P
   cues=(r"one of them|fit right in|true villager|one of the runners|fit in with|one of the family",
         r"odd one out|do not fit|outsider|stick out"),
   bases=[
    ("I work in a small design studio.",
     "Among my coworkers I feel like one of them.", "Among my coworkers I feel like the odd one out."),
    ("My partner's friends meet every weekend.",
     "When I am with them, I feel I fit right in.", "When I am with them, I feel I do not fit in at all."),
    ("I moved to a village with a strong local identity.",
     "I now feel like a true villager, just like everyone else here.", "I still feel like an outsider, different from everyone else here."),
    ("I joined a running club that meets at the sports shop.",
     "I feel I am one of the runners there, just like the rest.", "I feel I stick out among the runners there, unlike the rest."),
    ("I am the newest member of the nursing staff on my ward.",
     "I feel I fit in with the other nurses as if I had always been there.", "I feel I am the odd one out among the other nurses."),
    ("At gatherings on my partner's side, the conversation is lively.",
     "I feel like one of the family.", "I feel like an outsider in the family."),
   ]),
  "lasting_bonds": dict(material="new",  # P Pu C W P Pu
   cues=(r"stayed close|lasted|still close|remained|stayed strong", r"drifting apart|fizzling|fell apart|never last|breaking down|falling apart"),
   bases=[
    ("I have a group of friends from my school days.",
     "We have stayed close for over twenty years.", "We keep drifting apart and losing touch within months."),
    ("I have been in the town's walking group for a while.",
     "The friendships I made there have lasted for years.", "The friendships I made there keep fizzling out after a few weeks."),
    ("I met several people through a cooking class at a kitchen store.",
     "We are still close friends years later.", "Those friendships fell apart after a few weeks."),
    ("I have worked at the same company for a decade.",
     "My work friendships there have lasted all that time.", "My work friendships there never last more than a few months."),
    ("I have a few close friends from university.",
     "We have remained a steady part of each other's lives.", "Our friendships keep breaking down and starting over."),
    ("I sing in the church choir.",
     "The bonds I have with the other singers have stayed strong for years.", "The bonds I form with the other singers keep falling apart."),
   ]),
 },
 "satisfaction": {
  "quality": dict(material="new",  # Pu C W P Pu C
   cues=(r"correct and|neatly|good as new|perfectly|completely dry|clean, quiet|excellent",
         r"misspelled|crookedly|worse than|did not work|dripping|dirty|bland"),
   bases=[
    ("I renewed my passport at the city office.",
     "The new passport came out correct and well printed.", "The new passport came out with my name misspelled and smudged."),
    ("I had my winter coat repaired at a tailor.",
     "The seams were stitched neatly, and the coat looks as good as new.", "The seams were stitched crookedly, and the coat looks worse than before."),
    ("The IT department set up my new laptop.",
     "Everything worked perfectly from the first day.", "Half of the programs did not work on the first day."),
    ("A plumber fixed the leaking pipe in our kitchen.",
     "The pipe has been completely dry since then.", "The pipe has been dripping again since then."),
    ("I took the regional train to the coast.",
     "The carriage was clean, quiet and comfortable.", "The carriage was dirty, noisy and cramped."),
    ("I ordered a takeaway meal from a new restaurant.",
     "The food arrived hot and tasted excellent.", "The food arrived cold and tasted bland."),
   ]),
  "exceeded_expectations": dict(material="new",  # C W P Pu C W
   cues=(r"quicker than|more (comfortable|powerful|smoothly) than|better than|earlier than",
         r"slower than|less (comfortable|powerful|smoothly) than|worse than|later than"),
   bases=[
    ("My online order arrived in three days.",
     "That was quicker than the shop had promised.", "That was slower than the shop had promised."),
    ("The new office chairs arrived for our team.",
     "They were more comfortable than the catalogue had led me to expect.", "They were less comfortable than the catalogue had led me to expect."),
    ("The decorator painted our living room in two days.",
     "The finish looked better than I had expected from the samples.", "The finish looked worse than I had expected from the samples."),
    ("The council resurfaced the road outside my house.",
     "The work was finished earlier than the notice had said.", "The work was finished later than the notice had said."),
    ("I bought a vacuum cleaner on a friend's recommendation.",
     "It turned out to be more powerful than I had been led to believe.", "It turned out to be less powerful than I had been led to believe."),
    ("Our company switched to a new payroll provider.",
     "Their first month went more smoothly than we had been told to expect.", "Their first month went less smoothly than we had been told to expect."),
   ]),
  "value_for_money": dict(material="new",  # W P Pu C W P
   cues=(r"excellent value|more than we paid|well worth|bargain|good value|a steal",
         r"overpriced|less than we paid|not worth|rip-off|poor value|too expensive"),
   bases=[
    ("Our office ordered a new coffee machine.",
     "For what it cost, it was excellent value.", "For what it cost, it was badly overpriced."),
    ("We hired a cleaning service for the house.",
     "Given the price, we got far more than we paid for.", "Given the price, we got far less than we paid for."),
    ("I bought a season ticket for the city swimming pool.",
     "It was well worth the money.", "It was not worth the money at all."),
    ("I paid for a guided tour of the old harbour.",
     "For the ticket price, it was a bargain.", "For the ticket price, it was a rip-off."),
    ("My employer paid for an external training course for me.",
     "Considering the fee, the course was good value.", "Considering the fee, the course was poor value."),
    ("We bought a second-hand sofa from a furniture store.",
     "At that price, it was a steal.", "At that price, it was far too expensive."),
   ]),
 },
 "trust": {
  "ability": dict(material="new",  # P Pu C W P Pu
   cues=(r"knows what|highly skilled|experts|\bcompetent|\bcapable|\bexperienced and",
         r"no idea|poorly skilled|clueless|incompetent|hopeless|inexperienced"),
   bases=[
    ("We use the same family doctor for all our check-ups.",
     "She clearly knows what she is doing.", "She clearly has no idea what she is doing."),
    ("The city has a new building inspector for our district.",
     "He is highly skilled and catches every problem.", "He is poorly skilled and misses obvious problems."),
    ("I take my car to the same garage every year.",
     "The mechanics there are experts who diagnose problems quickly.", "The mechanics there are clueless and misdiagnose problems constantly."),
    ("Our department relies on an outside accounting firm.",
     "The accountants there are highly competent.", "The accountants there are thoroughly incompetent."),
    ("My neighbour often repairs things around our house.",
     "He is very capable with tools.", "He is hopeless with tools."),
    ("Our town's water supply is managed by a local utility.",
     "The engineers there are experienced and handle problems expertly.", "The engineers there are inexperienced and handle problems clumsily."),
   ]),
  "benevolence": dict(material="new",  # Pu C W P Pu C
   cues=(r"residents' interests|best for me|my development|genuinely cares|children's interests first|out for my interests",
         r"own interests|commission|only for herself|only about|own reputation|bank's profits"),
   bases=[
    ("Our local council member handles residents' complaints.",
     "She always looks out for residents' interests.", "She only looks out for her own interests."),
    ("I buy insurance through the same agent.",
     "He always recommends what is best for me, even when he earns less.", "He always recommends what earns him the biggest commission."),
    ("My manager decides who gets which projects.",
     "She looks out for my development and gives me projects that help me grow.",
     "She looks out only for herself and gives me projects that make her look good."),
    ("My landlord lives in the same building.",
     "He genuinely cares about how his tenants are doing.", "He cares only about collecting the rent."),
    ("Our school's head teacher meets parents every month.",
     "He clearly puts the children's interests first.", "He clearly puts his own reputation first."),
    ("I have used the same bank adviser for years.",
     "She always looks out for my interests.", "She always looks out for the bank's profits."),
   ]),
  "integrity": dict(material="new",  # C W P Pu C W
   cues=(r"\bkept\b|paid it back exactly|turned out to be true|honestly", r"broken|lied|gone back|false|misrepresents"),
   bases=[
    ("I have ordered from the same online seller several times.",
     "They have always kept their promises about delivery.", "They have repeatedly broken their promises about delivery."),
    ("Our director made commitments to the staff about pay this year.",
     "She has kept every one of them.", "She has broken every one of them."),
    ("My brother borrowed money from me last year.",
     "He paid it back exactly when he said he would.", "He lied about when he would pay it back."),
    ("Our mayor made several promises before the election.",
     "She has kept all of them.", "She has gone back on all of them."),
    ("The car dealer described the used car's history to me.",
     "Everything he told me turned out to be true.", "Much of what he told me turned out to be false."),
    ("My team lead reports our progress to upper management.",
     "He always reports it honestly.", "He routinely misrepresents it."),
   ]),
 },
 "fairness": {
  "distributive": dict(material="new",  # W P Pu C W P
   cues=(r"reflected|in proportion|matched", r"smaller than|did the least|contributed least|half the price|did nothing|pay the most"),
   bases=[
    ("I worked extra hours to finish our team's biggest project.",
     "My bonus reflected the extra work I put in.", "My bonus was smaller than those of colleagues who did far less."),
    ("My siblings and I shared the work of clearing out our parents' house.",
     "The furniture was divided in proportion to how much each of us did.", "The furniture went mostly to the sibling who did the least work."),
    ("Our street raised money for new playground equipment.",
     "Each household's say in the choice matched what it had contributed.", "The household that contributed least got to choose everything."),
    ("I returned a broken kettle to the shop.",
     "The refund matched the full price I had paid.", "The refund was half the price I had paid, while others got full refunds."),
    ("Our team split the commission from a large sale.",
     "Each person's share matched their part in closing the deal.", "The person who did nothing got the largest share."),
    ("My housemates and I split the household bills.",
     "Each of us pays in proportion to how much we use.", "I pay the most even though I use the least."),
   ]),
  "procedural": dict(material="new",  # P Pu C W P Pu
   cues=(r"same rotation|same published criteria|standard rules|same clear criteria|one vote|applied to all",
         r"decided alone|behind closed doors|changed the rules|who he liked|her friends|sit on the board"),
   bases=[
    ("Our family decided who would host the holidays this year.",
     "We used the same rotation as always, and everyone could give their view.", "My aunt decided alone, without listening to anyone."),
    ("The city chose which neighbourhoods get new bike lanes.",
     "It applied the same published criteria everywhere and held hearings for residents.",
     "It chose behind closed doors, and residents were never heard."),
    ("My insurance company reviewed my claim.",
     "They followed their standard rules and let me explain my side.", "They changed the rules partway through and never let me explain my side."),
    ("My company selected people for the new training program.",
     "Everyone was assessed against the same clear criteria.", "The manager picked people based on who he liked."),
    ("Our tenants' association voted on how to spend the repair fund.",
     "Every tenant had one vote, and the rules were set in advance.", "The chair counted only the votes of her friends."),
    ("The sports league decided which teams get the best pitch times.",
     "It used a lottery that applied to all teams equally.", "It gave the best times to the teams whose coaches sit on the board."),
   ]),
  "interpersonal": dict(material="new",  # Pu C W P Pu C
   cues=(r"politely|courteous|respectfully|dignity|kindly", r"rudely|mocked|insults|contempt|shouted|dismissively"),
   bases=[
    ("I went to the tax office to ask about a letter.",
     "The official spoke to me politely and with respect.", "The official spoke to me rudely and talked down to me."),
    ("I called customer service about a billing error.",
     "The agent treated me courteously throughout the call.", "The agent mocked my questions throughout the call."),
    ("My supervisor gave me feedback on my report.",
     "She delivered it respectfully and considerately.", "She delivered it with insults and sarcasm."),
    ("My in-laws discussed our wedding plans with us.",
     "They treated our views with dignity and respect.", "They treated our views with contempt and ridicule."),
    ("A police officer stopped me to check my bicycle lights.",
     "He spoke to me courteously the whole time.", "He shouted at me and belittled me the whole time."),
    ("I asked a pharmacist about the side effects of a medicine.",
     "She answered kindly and took my concern seriously.", "She answered dismissively and made fun of my concern."),
   ]),
 },
 "blame": {
  "agent_caused": dict(material="new",  # C W P Pu C W
   cues=(r"clerk|courier|teenager|driver|\bcook\b|operator", r"gust|rainwater|branch|lightning|contamination|earthquake"),
   bases=[
    ("A customer at the supermarket was hurt when a display collapsed onto her.",
     "A stock clerk had pushed a loaded trolley into the display.", "A sudden gust through the open doors had knocked the display over."),
    ("A visitor to our office slipped and broke his wrist in the lobby.",
     "A courier had spilled a drink on the floor right before.", "Rainwater had blown in through the door right before."),
    ("My neighbour's car windscreen was cracked overnight.",
     "A teenager had thrown a stone at it.", "A falling branch had hit it in the storm."),
    ("The town's war memorial was damaged last week.",
     "A driver had crashed a van into it.", "Lightning had struck it during a storm."),
    ("A diner at a restaurant got food poisoning.",
     "The cook had served chicken that he left out all night.", "Hidden contamination at the farm had spread to the chicken."),
    ("A shipment for a client arrived damaged at our warehouse.",
     "A forklift operator had driven into the pallets.", "An earthquake tremor had toppled the pallets."),
   ]),
  "preventable": dict(material="new",  # W P Pu C W P
   cues=(r"had been warned|known for months|had been told|ignored|after being reminded",
         r"as required|nobody could have known|no earlier sign|impossible to read|without any warning|broke suddenly"),
   bases=[
    ("A colleague's laptop with client data was stolen from his car.",
     "He had been warned many times not to leave it in the car and ignored the warnings.",
     "He had locked it in the boot as required, and the thieves cut through the lock."),
    ("My uncle's basement flooded and ruined the boxes I had stored there.",
     "He had known for months that the drain was blocked and did nothing.", "The drain had always worked, and nobody could have known it would fail."),
    ("A child was hurt on a broken swing in the public park.",
     "The park keeper had been told about the broken chain weeks earlier.", "The chain had snapped that morning with no earlier sign of damage."),
    ("My package was delivered to the wrong address and lost.",
     "The courier had been given the correct address and ignored it.", "The address label had been smudged in the rain and was impossible to read."),
    ("Our team's server went down for a day.",
     "The technician had been told the backup power was failing and never replaced it.",
     "The backup power failed without any warning that anyone could have seen."),
    ("My friend's dog got out and was hit by a bicycle.",
     "My friend had left the gate open again after being reminded.", "The gate latch broke suddenly while my friend was inside."),
   ]),
  "unjustified": dict(material="new",  # P Pu C W P Pu
   cues=(r"could not be bothered|finish his shift early|felt like|football match|did not want to work",
         r"injured|collapsed|gas leak|caught fire|burns|rockslide"),
   bases=[
    ("My neighbour drove across our lawn and left deep ruts.",
     "He did it because he could not be bothered to use the driveway.", "He did it to get an injured child to the hospital quickly."),
    ("A bus driver skipped our stop, and several of us missed our trains.",
     "He skipped it because he wanted to finish his shift early.", "He skipped it to get a passenger who had collapsed to an ambulance."),
    ("The pharmacy closed an hour early, and I could not get my medicine.",
     "The pharmacist closed early because she felt like going home.", "The pharmacist closed early because a gas leak forced everyone out."),
    ("My colleague missed the client meeting, and we lost the deal.",
     "He skipped it to go to a football match.", "He skipped it because his house had caught fire."),
    ("My housemate used up all the hot water before my job interview.",
     "She did it because she felt like taking a long bath.", "She did it to cool burns after an accident in the kitchen."),
    ("A park ranger closed the trail on the day of our charity walk.",
     "He closed it because he did not want to work that afternoon.", "He closed it because a rockslide had made it dangerous."),
   ]),
 },
 "intentionality": {
  "desire": dict(material="new",  # Pu C W P Pu C
   cues=(r"exactly (the|what)", r"did not actually want"),
   bases=[
    ("The librarian moved the reading group to the larger room.",
     "That was exactly the change she had been hoping for.", "That was a change she did not actually want."),
    ("The baker switched the shop's opening time to seven.",
     "That was exactly what he had been wanting for months.", "That was something he did not actually want."),
    ("My colleague ended up leading the new project.",
     "Leading it was exactly what she had been hoping for.", "Leading it was something she did not actually want."),
    ("My brother repainted his kitchen in pale yellow.",
     "Pale yellow was exactly the colour he had been hoping for.", "Pale yellow was a colour he did not actually want."),
    ("The council member moved the weekly market to Saturday.",
     "That was exactly the outcome she had been hoping for.", "That was an outcome she did not actually want."),
    ("The shop owner rearranged the shelves by colour.",
     "That was exactly the look he had been hoping for.", "That was a look he did not actually want."),
   ]),
  "planned": dict(material="new",  # C W P Pu C W
   cues=(r"planning|planned|decided weeks before", r"on a whim|on the spot|on impulse"),
   bases=[
    ("My neighbour bought a red bicycle last week.",
     "She had been planning the purchase for weeks.", "She bought it on a whim while passing the shop."),
    ("My manager rearranged the desks in our office.",
     "He had planned the new layout carefully in advance.", "He moved them around on a whim one afternoon."),
    ("My father cut his long hair short.",
     "He had decided weeks before and booked the appointment in advance.", "He decided on the spot while sitting in the barber's chair."),
    ("A man in the square started playing the violin for the crowd.",
     "He had planned the performance days ahead.", "He started on impulse when he saw the crowd."),
    ("My friend ordered the most expensive dish on the menu.",
     "She had planned to order it before we even arrived.", "She ordered it on a whim without looking at the price."),
    ("A colleague signed up our team for a charity run.",
     "He had planned it for months and checked everyone's calendars.", "He signed us up on impulse while scrolling through a website."),
   ]),
  "aware": dict(material="new",  # W P Pu C W P
   cues=(r"knew exactly|fully aware|full awareness|knowing exactly", r"absent-mindedly|without noticing|without realising"),
   bases=[
    ("My colleague sent the newsletter to the whole company.",
     "He knew exactly what he was doing when he clicked send.", "He clicked send absent-mindedly without noticing who it went to."),
    ("My sister switched off the lights in the hall.",
     "She was fully aware of what she was doing.", "She did it without noticing, while talking on the phone."),
    ("A man in the queue pressed the button for the next tram stop.",
     "He pressed it fully aware of what he was doing.", "He pressed it without realising, while leaning on the pole."),
    ("My friend added an extra coffee to our order.",
     "She knew exactly what she was doing when she added it.", "She added it absent-mindedly while reading her messages."),
    ("My coworker turned the office thermostat up a degree.",
     "He did it with full awareness of what he was doing.", "He did it without noticing, while leaning against the wall."),
    ("My partner set the alarm for six o'clock.",
     "He set it knowing exactly what he was doing.", "He set it absent-mindedly while half asleep."),
   ]),
 },
 "political_stance": {
  "economic": dict(material="new",  # P Pu C W P Pu
   cues=(r"cut taxes|lowering|keep more|left to the market|abolished|reduced", r"raise taxes|raising|pay more|raised by law|increased"),
   bases=[
    ("At a family dinner, my uncle asked what I think about taxes.",
     "I said the government should cut taxes on high earners and spend less on welfare.",
     "I said the government should raise taxes on high earners and spend more on welfare."),
    ("At a town hall meeting, residents discussed the city budget.",
     "I argued for lowering business taxes and trimming housing subsidies.", "I argued for raising business taxes and expanding housing subsidies."),
    ("While waiting at the bank, I chatted with another customer about the economy.",
     "I told him wealthy people should keep more of what they earn.", "I told him wealthy people should pay more to support the poor."),
    ("During lunch at work, colleagues talked about the minimum wage.",
     "I said the minimum wage should be left to the market.", "I said the minimum wage should be raised by law."),
    ("My partner and I discussed the new inheritance tax proposal.",
     "I told her that inheritance taxes should be abolished.", "I told her that inheritance taxes on large estates should be increased."),
    ("At a public debate, the topic was unemployment benefits.",
     "I argued that benefits should be reduced to encourage work.", "I argued that benefits should be increased to protect people out of work."),
   ]),
  "social_change": dict(material="new",  # Pu C W P Pu C
   cues=(r"tradition|old customs", r"\bchange\b|changing|changed|reinvented|reconsidered|modern"),
   bases=[
    ("At a school board meeting, parents debated the curriculum.",
     "I argued that schools should stick to traditional values.", "I argued that schools should update their teaching to reflect a changing society."),
    ("At the bookshop, the owner asked my view on a new book about family life.",
     "I said families work best when they keep to traditional roles.", "I said families work best when roles change with the times."),
    ("At a work event, colleagues discussed national holidays.",
     "I said we should keep the holidays exactly as tradition has them.", "I said we should change the holidays to reflect how the country has changed."),
    ("My grandmother asked what I think about modern weddings.",
     "I told her weddings should follow the old customs.", "I told her weddings should be reinvented for today's couples."),
    ("At a community forum, people discussed the town's historic statues.",
     "I said they should stay exactly where tradition placed them.", "I said they should be reconsidered as the community changes."),
    ("In a café, a stranger asked my opinion on dress codes.",
     "I said traditional dress codes should be kept.", "I said dress codes should change with modern norms."),
   ]),
  "criminal_justice": dict(material="new",  # C W P Pu C W
   cues=(r"longer prison|tougher penalties|strict punishment|harsher|more severely|on punishment",
         r"job training|rehabilitation|second chance|shorter sentences|health problems"),
   bases=[
    ("At the hardware store, the owner and I talked about recent thefts.",
     "I said thieves should face much longer prison sentences.", "I said thieves should get job training and treatment instead of prison."),
    ("At lunch, my colleagues discussed a proposed crime bill.",
     "I said it should impose tougher penalties on repeat offenders.", "I said it should fund rehabilitation programs for repeat offenders."),
    ("My brother asked what I think about juvenile crime.",
     "I told him young offenders need strict punishment.", "I told him young offenders need counselling and a second chance."),
    ("At a neighbourhood meeting, residents discussed policing.",
     "I argued for more police patrols and harsher sentences.", "I argued for more youth programs and shorter sentences."),
    ("At the barber's, the conversation turned to drug laws.",
     "I said drug offences should be punished more severely.", "I said drug offences should be treated as health problems."),
    ("In a work meeting about community projects, prisons came up.",
     "I said prisons should focus on punishment.", "I said prisons should focus on rehabilitation."),
   ]),
 },
}


# ------------------------------------------------------------------ registry

def load_registry(path=REGISTRY):
    import yaml
    return yaml.safe_load(open(path))


def file_sha256(path=REGISTRY):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def indicator_list(reg):
    """[(construct, indicator)] in registry order (global indicator index k)."""
    return [(c, i) for c in reg["rating_order"] for i in reg["constructs"][c]["indicators"]]


def expected_matrix(reg):
    """Signed expected matrix E[rating, (construct.indicator)]: +-2, +-1, 0, NaN for NA."""
    ratings = reg["rating_order"]
    cols = [f"{c}.{i}" for c, i in indicator_list(reg)]
    E = pd.DataFrame(0.0, index=ratings, columns=cols)
    for c, i in indicator_list(reg):
        ind = reg["constructs"][c]["indicators"][i]
        E.loc[c, f"{c}.{i}"] = 2.0
        for r, v in (ind.get("expected") or {}).items():
            assert r in ratings and r != c, (c, i, r)
            E.loc[r, f"{c}.{i}"] = np.nan if v in (None, "NA") else float(v)
    return E


def expected_sha256(reg):
    E = expected_matrix(reg)
    canon = json.dumps({"rows": list(E.index), "cols": list(E.columns),
                        "values": [[None if np.isnan(x) else float(x) for x in row] for row in E.values]},
                       sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()


# ------------------------------------------------------------------ generation

def generate(reg=None):
    reg = reg or load_registry()
    rows = []
    for k, (c, i) in enumerate(indicator_list(reg)):
        spec = STIM[c][i]
        for b, (ctx, hi, lo) in enumerate(spec["bases"]):
            domain = DOMAINS[(k % 4 + b) % 4]
            filler = FILLER[(k * 6 + b) % len(FILLER)]
            base_id = f"{c}.{i}.b{b}"
            for pole, ev in [("high", hi), ("low", lo)]:
                rows.append(dict(id=f"{base_id}.{pole}", base_id=base_id, construct=c, indicator=i,
                                 col=f"{c}.{i}", family=reg["constructs"][c]["family"], domain=domain, pole=pole,
                                 material=spec["material"], text=" ".join([ctx, ev, filler]),
                                 evidence=ev, context=ctx))
    return pd.DataFrame(rows)


def stimuli_sha256(df):
    return hashlib.sha256("\n".join(df.sort_values("id").apply(lambda r: f"{r.id}\t{r.text}", axis=1)).encode()).hexdigest()


def reversed_subset(df):
    """Balanced 25% subset of base pairs for the reversed-anchor condition: per construct, indicators get 2,1,2 or
    1,2,1 bases (alternating by construct), i.e. 4 or 5 per construct, 54 of 216 pairs, both poles of each."""
    keep = []
    for ci, (c, g) in enumerate(df.groupby("construct", sort=False)):
        pattern = [2, 1, 2] if ci % 2 == 0 else [1, 2, 1]
        for n, (ind, gi) in zip(pattern, g.groupby("indicator", sort=False)):
            keep += sorted(gi.base_id.unique())[:n]
    return sorted(keep)


# ------------------------------------------------------------------ checks

SENT = re.compile(r"(?<=[.!?])\s+")
STOP = set("a an the and or of to in on at for with by from as is was were be been has have had it its this that "
           "i me my we our you your he she they them his her their there here then than so but not no".split())


def content_words(s):
    return {w for w in re.findall(r"[a-z']+", s.lower()) if w not in STOP and len(w) > 2}


def checks(df, reg=None):
    reg = reg or load_registry()
    out = []
    ind = indicator_list(reg)
    cell = df.groupby(["construct", "indicator", "pole"]).size()
    out.append(("12 constructs x 3 indicators present", len(ind) == 36 and set(df.col) == {f"{c}.{i}" for c, i in ind},
                f"{df.col.nunique()} indicators"))
    out.append(("exact cell balance (6 per indicator x pole)", cell.nunique() == 1 and cell.iloc[0] == 6 and len(cell) == 72,
                f"{len(cell)} cells, sizes {sorted(cell.unique())}"))
    pairs = df.groupby("base_id").pole.apply(lambda s: sorted(s))
    out.append(("complete high/low pairing", all(p == ["high", "low"] for p in pairs), f"{len(pairs)} base pairs"))
    out.append(("unique ids", df.id.is_unique, f"{df.id.nunique()} ids"))
    out.append(("no duplicate texts", df.text.is_unique, f"{df.text.nunique()} unique / {len(df)}"))
    pair_same_ctx = df.groupby("base_id").apply(lambda g: g.context.nunique() == 1 and g.evidence.nunique() == 2,
                                                include_groups=False)
    out.append(("pairs share context, differ in evidence", bool(pair_same_ctx.all()), f"{int((~pair_same_ctx).sum())} violations"))
    bad = [(r.id, p) for r in df.itertuples() for p in FORBIDDEN if re.search(p, r.text, re.I)]
    out.append(("no forbidden target words", not bad, f"{len(bad)} hits" + (f": {bad[:4]}" if bad else "")))
    ns = df.text.map(lambda t: len(SENT.split(t.strip())))
    out.append(("equal sentence counts", ns.nunique() == 1, f"sentences per text: {sorted(ns.unique())}"))
    # length balance: within each pair and between poles overall
    lens = df.pivot_table(index="base_id", columns="pole", values="evidence", aggfunc=lambda s: len(s.iloc[0]))
    rel = (lens.high - lens.low).abs() / lens[["high", "low"]].max(axis=1)
    overall = abs(lens.high.mean() - lens.low.mean()) / lens.values.mean()
    out.append(("length balance (pair diff <= 60%, pole means within 10%)", bool(rel.max() <= .60 and overall <= .10),
                f"max pair diff {rel.max():.0%}, pole mean diff {overall:.1%}"))
    # polarity: high evidence matches a high cue that the low evidence lacks; low evidence matches a low cue
    pol = []
    for (c, i), g in df.groupby(["construct", "indicator"]):
        hi_re, lo_re = STIM[c][i]["cues"]
        for b, gb in g.groupby("base_id"):
            h, l = gb[gb.pole == "high"].evidence.iloc[0], gb[gb.pole == "low"].evidence.iloc[0]
            if not (re.search(hi_re, h, re.I) and not re.search(hi_re, l, re.I) and re.search(lo_re, l, re.I)):
                pol.append(b)
    out.append(("polarity cues consistent (no reversed pair)", not pol, f"{len(pol)} suspicious" + (f": {pol[:4]}" if pol else "")))
    # metadata leakage: labels must not reach the model. Construct names are covered by FORBIDDEN; here: pole labels and
    # multi-word indicator identifiers (single plain words such as "accepted" are the concrete evidence itself, by design).
    labels = {"high pole", "low pole", "political stance"} | {i.replace("_", " ") for _, i in ind if "_" in i} | \
             {i for _, i in ind if "_" in i}
    leak = [r.id for r in df.itertuples() if any(re.search(rf"\b{re.escape(t)}\b", r.text, re.I) for t in labels)]
    out.append(("no metadata labels in texts", not leak, f"{len(leak)} hits" + (f": {leak[:4]}" if leak else "")))
    # lexical overlap between evidence vocabularies of indicators in different families
    vocab = {col: set().union(*df[df.col == col].evidence.map(content_words)) for col in df.col.unique()}
    fam = dict(zip(df.col, df.family))
    worst = max(((len(vocab[a] & vocab[b]) / len(vocab[a] | vocab[b]), a, b)
                 for a in vocab for b in vocab if a < b and fam[a] != fam[b]))
    out.append(("cross-family evidence vocabulary overlap (Jaccard) < 0.25", worst[0] < .25,
                f"max {worst[0]:.2f} ({worst[1]} vs {worst[2]})"))
    dom = df.drop_duplicates("base_id").domain.value_counts()
    per_ind = df.drop_duplicates("base_id").groupby("col").domain.nunique()
    out.append(("domain balance (54 pairs per domain; every indicator spans 4)", dom.nunique() == 1 and dom.iloc[0] == 54
                and per_ind.min() == 4, dom.to_dict()))
    rs = reversed_subset(df)
    out.append(("reversed-anchor subset is 25% and spans all indicators",
                len(rs) == 54 and df[df.base_id.isin(rs)].col.nunique() == 36, f"{len(rs)} pairs"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    reg = load_registry()
    df = generate(reg)
    res = checks(df, reg)
    for name, ok, detail in res:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\n{len(df)} texts, {df.base_id.nunique()} pairs; registry sha256 {file_sha256()[:12]}, "
          f"expected-matrix sha256 {expected_sha256(reg)[:12]}, stimuli sha256 {stimuli_sha256(df)[:12]}")
    if a.out:
        df.to_csv(a.out, index=False)
    sys.exit(0 if all(ok for _, ok, _ in res) else 1)


if __name__ == "__main__":
    main()
