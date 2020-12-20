use baseball;

CREATE OR REPLACE TABLE bc_join_g 
SELECT batter,atBat , Hit , b.game_id , local_date
FROM batter_counts b JOIN game g ON b.game_id = g.game_id 
ORDER BY batter, b.game_id ,local_date;

select * from atbat_r ar;
select * from batter_counts bc;
select * from batter b;
select * from battersInGame big;
select * from bc_join_g bjg;
select * from book b;
select * from boxscore b;
select * from game g;
select * from game_inv_dtl gid ;
select * from game_temp gt ;
select * from hits h ;
select * from inning i ;
select * from inning_action ia ;
select * from  league_division_team ldt ;
select * from linescore l ;
select * from lineup l ;
select * from odds_hist oh ;
select * from pitcher_counts pc ;
select * from pitcher_stat ps ;
select * from pitchersInGame pig ;
select * from `position` p ;
select * from pregame p ;
select * from pregame_bet pb ;
select * from pregame_detail pd ;
select * from pregame_odds po ;
select * from stadium s ;
select * from team t ;
select * from team_batting_counts tbc ;
select * from team_game_prior_next tgpn ;
select * from team_pitching_counts tpc ;
select * from team_results tr ;
select * from team_streak ts ;



drop table if exists RollingAvgTable;
create table RollingAvgTable as
select
		tbc1.team_id
		,tbc1.game_id
		,g1.local_date
		,count(*) as cnt
		,sum(tbc2.plateApperance) as plateAppearance
		,sum(tbc2.atBat) as atBat
		,sum(tbc2.Hit)as Hit
		,sum(tbc2.caughtStealing2B)as caughtStealing2B
		,sum(tbc2.caughtStealing3B)as caughtStealing3B 
		,sum(tbc2.caughtStealingHome)as caughtStealingHome 
		,sum(tbc2.stolenBase2B)as stolenBase2B 
		,sum(tbc2.stolenBase3B)as stolenBase3B 
		,sum(tbc2.stolenBaseHome)as stolenBaseHome 
		,sum(tbc2.toBase)as toBase 
		,sum(tbc2.Batter_Interference)as Batter_Interference 
		,sum(tbc2.Bunt_Ground_Out)+sum(tbc2.Bunt_Groundout)as Bunt_Ground_Out 
		,sum(tbc2.Bunt_Pop_Out)as Bunt_Pop_Out 
		,sum(tbc2.Catcher_Interference)as Catcher_Interference 
		,sum(tbc2.`Double`)as `Double` 
		,sum(tbc2.Double_Play)as Double_Play 
		,sum(tbc2.Fan_interference)as Fan_interference 
		,sum(tbc2.Field_Error)as Field_Error 
		,sum(tbc2.Fielders_Choice)as Fielders_Choice 
		,sum(tbc2.Fly_Out)+sum(tbc2.Flyout) as Fly_Out 
		,sum(tbc2.Force_Out)+SUM(tbc2.Forceout)as Force_Out 
		,sum(tbc2.Ground_Out)+SUM(tbc2.Groundout)as Ground_Out 
		,sum(tbc2.Grounded_Into_DP)as Grounded_Into_DP 
		,sum(tbc2.Hit_By_Pitch)as Hit_By_Pitch 
		,sum(tbc2.Home_Run)as Home_Run 
		,sum(tbc2.Intent_Walk)as Intent_Walk 
		,SUM(tbc2.Line_Out)as Line_Out 
		,SUM(tbc2.Pop_Out)as Pop_Out 
		,SUM(tbc2.Runner_Out)as Runner_Out 
		,sum(tbc2.Sac_Bunt)as Sac_Bunt 
		,sum(tbc2.Sac_Fly)as Sac_Fly 
		,sum(tbc2.Sac_Fly_DP)as Sac_Fly_DP 
		,sum(tbc2.Sacrifice_Bunt_DP)as Sacrifice_Bunt_DP 
		,SUM(tbc2.Single)as Single
		,SUM(tbc2.Strikeout) as Strikeout 
		,SUM(tbc2.`Strikeout_-_DP`)as `Strikeout_-_DP` 
		,SUM(tbc2.`Strikeout_-_TP`)as `Strikeout_-_TP` 
		,sum(tbc2.Triple)as Triple 
		,sum(tbc2.Triple_Play)as Triple_Play 
		,sum(tbc2.Walk)as walk
FROM team_batting_counts tbc1 
join game g1 on tbc1.game_id = g1.game_id and g1.`type` IN ("R")
join team_batting_counts tbc2 on tbc1.team_id = tbc2.team_id 
join game g2 on tbc2.game_id = g2.game_id and g2.`type` in ("R")
and g2.local_date < g1.local_date and 
g2.local_date >= DATE_ADD(g1.local_date,interval -200 day)
group by tbc1.team_id,tbc1.game_id,g1.local_date
order by g1.local_date,tbc1.team_id;
create unique index team_game on RollingAvgTable(team_id,game_id);
select * from RollingAvgTable;

#fixing stolenbase calculations
drop table fix_stealing ;
create table fix_stealing as 
select * from 
	(select g.game_id ,
	 g.away_team_id as team_id,
	 sum(case when des = "Stolen Base 2B" then 1 else 0 end) as stolenBase2B,
	 sum(case when des = "Stolen Base 3b" then 1 else 0 end) as stolenBase3B,
	 sum(case when des = "Stolen Base Home" then 1 else 0 end) as stolenBaseHome,
	 sum(case when des = "Caught Stealing 2b"then 1 else 0 end) as caughtStealing2B,
	 sum(case when des = "Caught Stealing 3b"then 1 else 0 end) as caughtStealing3B,
	 sum(case when des = "Caught Stealing Home"then 1 else 0 end) as caughtStealingHome
	 from inning i
	 join game g
	 on i.game_id = g.game_id 
	 where i.half = 0 and i.entry = "runner"
	 group by g.game_id , g.away_team_id 
	 UNION 
	 select g.game_id ,
	 g.home_team_id as team_id,
	 sum(case when des = "Stolen Base 2B" then 1 else 0 end) as stolenBase2B,
	 sum(case when des = "Stolen Base 3b" then 1 else 0 end) as stolenBase3B,
	 sum(case when des = "Stolen Base Home" then 1 else 0 end) as stolenBaseHome,
	 sum(case when des = "Caught Stealing 2b"then 1 else 0 end) as caughtStealing2B,
	 sum(case when des = "Caught Stealing 3b"then 1 else 0 end) as caughtStealing3B,
	 sum(case when des = "Caught Stealing Home"then 1 else 0 end) as caughtStealingHome
	 from inning i
	 join game g
	 on i.game_id = g.game_id 
	 where i.half = 0 and i.entry = "runner"
	 group by g.game_id , g.home_team_id  
	 ) as subTable
	order by game_id,team_id;	
drop index team_game_uidx on fix_stealing;
create unique index team_game_uidx on fix_stealing(team_id,game_id);

#fixing team batting counts
drop table if exists team_batting_counts_fixed;
create table team_batting_counts_fixed like team_batting_counts;
drop index team_game_uidx on team_batting_counts_fixed;
create unique index team_game_uidx on team_batting_counts_fixed(team_id,game_id);
insert into team_batting_counts_fixed select*from team_batting_counts ;

insert into team_batting_counts_fixed(game_id,team_id,stolenBase2B,stolenBase3B,stolenBaseHome,caughtStealing2B,caughtStealing3B,caughtStealingHome)
select game_id,team_id,stolenBase2B,stolenBase3B,stolenBaseHome,caughtStealing2B,caughtStealing3B,caughtStealingHome from fix_stealing fs
on duplicate key update
stolenBase2B = fs.stolenBase2B,
stolenBase3B = fs.stolenBase3B,
stolenBaseHome = fs.stolenBaseHome,
caughtStealing2B = fs.caughtStealing2B,
caughtStealing3B = fs.caughtStealing3B,
caughtStealingHome = fs.caughtStealingHome;

select * from team_batting_counts tbc ;
select * from team_batting_counts_fixed;

drop table if exists RollingAvgTable;

create table RollingAvgTable as
select
		tbc1.team_id,tbc1.game_id,g1.local_date,count(*) as cnt
		,sum(tbc2.plateApperance) as plateAppearance
		,sum(tbc2.atBat) as atBat ,sum(tbc2.Hit)as Hit
		,sum(tbc2.caughtStealing2B)as caughtStealing2B ,sum(tbc2.caughtStealing3B)as caughtStealing3B ,sum(tbc2.caughtStealingHome)as caughtStealingHome 
		,sum(tbc2.stolenBase2B)as stolenBase2B ,sum(tbc2.stolenBase3B)as stolenBase3B 
		,sum(tbc2.stolenBaseHome)as stolenBaseHome 
		,sum(tbc2.toBase)as toBase ,sum(tbc2.Batter_Interference)as Batter_Interference 
		,sum(tbc2.Bunt_Ground_Out)+sum(tbc2.Bunt_Groundout)as Bunt_Ground_Out 
		,sum(tbc2.Bunt_Pop_Out)as Bunt_Pop_Out 
		,sum(tbc2.Catcher_Interference)as Catcher_Interference 
		,sum(tbc2.`Double`)as `Double` ,sum(tbc2.Double_Play)as Double_Play 
		,sum(tbc2.Fan_interference)as Fan_interference 
		,sum(tbc2.Field_Error)as Field_Error ,sum(tbc2.Fielders_Choice)as Fielders_Choice 
		,sum(tbc2.Fly_Out)+sum(tbc2.Flyout) as Fly_Out 
		,sum(tbc2.Force_Out)+SUM(tbc2.Forceout)as Force_Out 
		,sum(tbc2.Ground_Out)+SUM(tbc2.Groundout)as Ground_Out 
		,sum(tbc2.Grounded_Into_DP)as Grounded_Into_DP 
		,sum(tbc2.Hit_By_Pitch)as Hit_By_Pitch 
		,sum(tbc2.Home_Run)as Home_Run ,sum(tbc2.Intent_Walk)as Intent_Walk 
		,SUM(tbc2.Line_Out)as Line_Out 
		,SUM(tbc2.Pop_Out)as Pop_Out ,SUM(tbc2.Runner_Out)as Runner_Out 
		,sum(tbc2.Sac_Bunt)as Sac_Bunt ,sum(tbc2.Sac_Fly)as Sac_Fly ,sum(tbc2.Sac_Fly_DP)as Sac_Fly_DP 
		,sum(tbc2.Sacrifice_Bunt_DP)as Sacrifice_Bunt_DP 
		,SUM(tbc2.Single)as Single,SUM(tbc2.Strikeout) as Strikeout 
		,SUM(tbc2.`Strikeout_-_DP`)as `Strikeout_-_DP` ,SUM(tbc2.`Strikeout_-_TP`)as `Strikeout_-_TP` 
		,sum(tbc2.Triple)as Triple 
		,sum(tbc2.Triple_Play)as Triple_Play 
		,sum(tbc2.Walk)as walk
FROM team_batting_counts_fixed tbc1 
join game g1 on tbc1.game_id = g1.game_id and g1.`type` IN ("R")
join team_batting_counts_fixed tbc2 on tbc1.team_id = tbc2.team_id 
join game g2 on tbc2.game_id = g2.game_id and g2.`type` in ("R")
and g2.local_date < g1.local_date and 
g2.local_date >= DATE_ADD(g1.local_date,interval -200 day)
group by tbc1.team_id,tbc1.game_id,g1.local_date
order by g1.local_date,tbc1.team_id;
create unique index team_game on RollingAvgTable(team_id,game_id);


select * from RollingAvgTable;

#RBA
select team_id, game_id, Hit/atBat as BA 
from RollingAvgTable;

select g.home_team_id,g.away_team_id, g.game_id, r2da.Hit/r2da.atBat as BA_Away, r2dh.Hit/r2dh.atBat as BA_home
from game g join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id
join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id;

select g.home_team_id,g.away_team_id, g.game_id, (r2dh.Hit/r2dh.atBat) /(r2da.Hit/r2da.atBat) -1.0 as BA_Ratio, (r2dh.Hit/r2dh.atBat) - (r2da.Hit/r2da.atBat) as BA_diff
from game g join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id
join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id;


#slugging percentage
select g.home_team_id,g.away_team_id, g.game_id, r2da.Hit/r2da.atBat as BA_Away, r2dh.Hit/r2dh.atBat as BA_home
, (r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )/(r2dh.atBat) as Slug_home
,(r2da.Single+2*r2da.`Double` +3*r2da.Triple+4*r2da.Home_Run)/(r2da.atBat) as Slug_away
from game g join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id
join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id;

select g.home_team_id,g.away_team_id, g.game_id, r2da.Hit/r2da.atBat as BA_Away, r2dh.Hit/r2dh.atBat as BA_home
, (r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )/(r2dh.atBat)-(r2da.Single+2*r2da.`Double` +3*r2da.Triple+4*r2da.Home_Run)/(r2da.atBat) as Slug_diff
,((r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )/(r2dh.atBat))/((r2da.Single+2*r2da.`Double` +3*r2da.Triple+4*r2da.Home_Run)/(r2da.atBat)) -1.0 as Slug_ratio
from game g join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id
join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id;

#Response Home team wins

select g.home_team_id,g.away_team_id, g.game_id, r2da.Hit/r2da.atBat as BA_Away, r2dh.Hit/r2dh.atBat as BA_home
, case when b.away_runs < b.home_runs then 1
	   when b.away_runs > b.home_runs then 0 
	   else 0 end as Home_team_wins
from game g join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id
join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join boxscore b on b.game_id = g.game_id ;


# Total Bases
select * from RollingAvgTable
select g.home_team_id, g.away_team_id, g.game_id, (r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run ) as Total_Bases_home, 
(r2da.Single+2*r2da.`Double` +3*r2da.Triple+4*r2da.Home_Run)as Total_Bases_away 
from game g join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id
join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id;


select ((r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )-(r2da.Single+2*r2da.`Double` +3*r2da.Triple+4*r2da.Home_Run)) as total_bases_diff,
((r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )/(r2da.Single+2*r2da.`Double` +3*r2da.Triple+4*r2da.Home_Run))- 1.0 as total_bases_ratio
from game g join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id
join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id;


# Extra bases on long hits
select g.home_team_id, g.away_team_id, g.game_id, (r2dh.`Double` +r2dh.Triple + r2dh.Home_Run ) as Extra_Bases_home, 
(r2da.`Double` +r2da.Triple+r2da.Home_Run)as Extra_Bases_away 
from game g join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id
join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id;

select ((r2dh.`Double` +r2dh.Triple + r2dh.Home_Run )-(r2da.`Double` +r2da.Triple+r2da.Home_Run)) as extra_bases_lh_diff,
((r2dh.`Double` +r2dh.Triple + r2dh.Home_Run )/(r2da.`Double` +r2da.Triple+r2da.Home_Run)) - 1.0 as extra_bases_li_ratio
from game g join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id
join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id;


# Total Average
select g.home_team_id, g.game_id, 
r2dh.Hit_By_Pitch, r2dh.walk, r2dh.stolenBaseHome, r2dh.caughtStealingHome, r2dh.atBat, r2dh.Hit, r2dh.Grounded_Into_DP,
(((r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )+ r2dh.Hit_By_Pitch+ r2dh.stolenBaseHome + r2dh.walk)/((r2dh.atBat- r2dh.Hit)
+ r2dh.caughtStealingHome+ r2dh.Grounded_Into_Dp)) as Total_Average
from game g join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id; 

# Times on Bases
select g.home_team_id, g.game_id,g.away_team_id, (r2dh.Hit+ r2dh.walk+ r2dh.Hit_By_Pitch) as Times_On_Bases_home,
(r2da.Hit+ r2da.walk+ r2da.Hit_By_Pitch) as Times_On_Bases_away
from game g join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id; 

select ((r2dh.Hit+ r2dh.walk+ r2dh.Hit_By_Pitch)-(r2da.Hit+ r2da.walk+ r2da.Hit_By_Pitch)) times_on_bases_diff,
((r2dh.Hit+ r2dh.walk+ r2dh.Hit_By_Pitch)/(r2da.Hit+ r2da.walk+ r2da.Hit_By_Pitch))- 1.0 as times_on_bases_ratio
from game g join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id; 

# On- Base Percentage
select g.home_team_id, g.game_id,g.away_team_id, ((r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch)/(r2dh.atBat+ r2dh.Walk+ r2dh.Hit_By_Pitch+ 
r2dh.Sac_Fly)) as on_base_percentage_home,((r2da.Hit+ r2da.Walk + r2da.Hit_By_Pitch)/(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly))
as on_base_percentage_away
from game g join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id;

select (((r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch)/(r2dh.atBat+ r2dh.Walk+ r2dh.Hit_By_Pitch+ 
r2dh.Sac_Fly))- ((r2da.Hit+ r2da.Walk + r2da.Hit_By_Pitch)/(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly))) as on_base_diff,
(((r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch)/(r2dh.atBat+ r2dh.Walk+ r2dh.Hit_By_Pitch+ r2dh.Sac_Fly))/ 
((r2da.Hit+ r2da.Walk + r2da.Hit_By_Pitch)/(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly))) -1.0 as on_base_ration
from game g join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id;


# on-base plus Slugging, Gross Production Average
select g.home_team_id, g.game_id, g.away_team_id, (((r2dh.atBat*(r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch))+ ((r2dh.Single + 2*r2dh.`Double` 
+3*r2dh.Triple + 4*r2dh.Home_Run )*(r2dh.atBat+ r2dh.Walk+ r2dh.Hit_By_Pitch+ r2dh.Sac_Fly)))/(r2dh.atBat * (r2dh.atBat+ r2dh.Walk+ 
r2dh.Hit_By_Pitch+ r2dh.Sac_Fly)))as on_base_plus_slugging_home, (((r2da.atBat*(r2da.Hit+ r2da.Walk + r2da.Hit_By_Pitch))+ ((r2da.Single + 2*r2da.`Double` 
+3*r2da.Triple + 4*r2dh.Home_Run )*(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly)))/(r2da.atBat * (r2da.atBat+ r2da.Walk+ 
r2da.Hit_By_Pitch+ r2da.Sac_Fly)))as on_base_plus_slugging_away,
((1.8*((r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch)/(r2dh.atBat+ r2dh.Walk+ 
r2dh.Hit_By_Pitch+ r2dh.Sac_Fly))+(r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )/(r2dh.atBat))/4) as gpa_home,
((1.8*((r2da.Hit+ r2da.Walk + r2da.Hit_By_Pitch)/(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly))+(r2da.Single + 
2*r2da.`Double` +3*r2da.Triple + 4*r2da.Home_Run )/(r2da.atBat))/4) as gpa_away
from game g join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id;

select ((((r2dh.atBat*(r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch))+ ((r2dh.Single + 2*r2dh.`Double` 
+3*r2dh.Triple + 4*r2dh.Home_Run )*(r2dh.atBat+ r2dh.Walk+ r2dh.Hit_By_Pitch+ r2dh.Sac_Fly)))/(r2dh.atBat * (r2dh.atBat+ r2dh.Walk+ 
r2dh.Hit_By_Pitch+ r2dh.Sac_Fly)))-(((r2da.atBat*(r2da.Hit+ r2da.Walk + r2da.Hit_By_Pitch))+ ((r2da.Single + 2*r2da.`Double` 
+3*r2da.Triple + 4*r2dh.Home_Run )*(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly)))/(r2da.atBat * (r2da.atBat+ r2da.Walk+ 
r2da.Hit_By_Pitch+ r2da.Sac_Fly)))) as on_base_slug_diff,
((((r2dh.atBat*(r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch))+ ((r2dh.Single + 2*r2dh.`Double` 
+3*r2dh.Triple + 4*r2dh.Home_Run )*(r2dh.atBat+ r2dh.Walk+ r2dh.Hit_By_Pitch+ r2dh.Sac_Fly)))/(r2dh.atBat * (r2dh.atBat+ r2dh.Walk+ 
r2dh.Hit_By_Pitch+ r2dh.Sac_Fly)))/(((r2da.atBat*(r2da.Hit+ r2da.Walk + r2da.Hit_By_Pitch))+ ((r2da.Single + 2*r2da.`Double` 
+3*r2da.Triple + 4*r2dh.Home_Run )*(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly)))/(r2da.atBat * (r2da.atBat+ r2da.Walk+ 
r2da.Hit_By_Pitch+ r2da.Sac_Fly))))-1.0 as on_base_slug_ratio
from game g join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id;

select (((1.8*((r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch)/(r2dh.atBat+ r2dh.Walk+ 
r2dh.Hit_By_Pitch+ r2dh.Sac_Fly))+(r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )/(r2dh.atBat))/4)-((1.8*((r2da.Hit+ 
r2da.Walk + r2da.Hit_By_Pitch)/(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly))+(r2da.Single + 2*r2da.`Double` +3*r2da.Triple + 
4*r2da.Home_Run )/(r2da.atBat))/4)) as gpa_diff,
(((1.8*((r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch)/(r2dh.atBat+ r2dh.Walk+ 
r2dh.Hit_By_Pitch+ r2dh.Sac_Fly))+(r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )/(r2dh.atBat))/4)/((1.8*((r2da.Hit+ 
r2da.Walk + r2da.Hit_By_Pitch)/(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly))+(r2da.Single + 2*r2da.`Double` +3*r2da.Triple + 
4*r2da.Home_Run )/(r2da.atBat))/4)) -1.0 as gpa_ratio
from game g join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id;


# at-bats per home run # Home runs per Hit
select g.home_team_id, g.game_id, g.away_team_id, (r2dh.atBat/r2dh.Home_Run) as at_bats_perHR_home,(r2da.atBat/r2da.Home_Run) as at_bats_perHR_away,
(r2dh.Home_Run/r2dh.Hit) as hr_h_home,(r2da.Home_Run/r2da.Hit) as hr_h_away
from game g join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id;

select ((r2dh.atBat/r2dh.Home_Run)- (r2da.atBat/r2da.Home_Run)) as at_bats_perHR_diff,
((r2dh.atBat/r2dh.Home_Run)/(r2da.atBat/r2da.Home_Run))- 1.0 as at_bats_perHR_ratio,
((r2dh.Home_Run/r2dh.Hit)-(r2da.Home_Run/r2da.Hit)) as hr_h_diff,
((r2dh.Home_Run/r2dh.Hit)/(r2da.Home_Run/r2da.Hit)) -1.0 as hr_h_ration
from game g join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id;


#  Plate appearances per strikeout
select g.home_team_id, g.game_id, g.away_team_id, (r2dh.plateAppearance/r2dh.Strikeout) as PA_per_SO_home,
(r2da.plateAppearance/r2da.Strikeout) as PA_per_SO_away
from game g join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id;

select ((r2dh.plateAppearance/r2dh.Strikeout)-(r2da.plateAppearance/r2da.Strikeout)) as PA_per_SO_diff,
((r2dh.plateAppearance/r2dh.Strikeout)/(r2da.plateAppearance/r2da.Strikeout)) -1.0 as PA_per_SO_ratio
from game g join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id;

#  Walk to Strike out Ratio
select g.home_team_id, g.game_id, g.away_team_id, (r2dh.walk/r2dh.Strikeout) as walk_SO_home,
(r2da.walk/r2da.Strikeout) as walk_SO_away
from game g join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id;

select ((r2dh.walk/r2dh.Strikeout)-(r2da.walk/r2da.Strikeout)) as walk_SO_diff,
((r2dh.walk/r2dh.Strikeout)/(r2da.walk/r2da.Strikeout)) as walk_SO_ratio
from game g join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id;

# Stolen Base percentage home
select g.home_team_id, g.game_id, g.away_team_id, (r2dh.stolenbasehome/(r2dh.stolenbasehome+ r2dh.caughtstealinghome)) as SB_percent_home ,
(r2da.stolenbasehome/(r2da.stolenbasehome+ r2da.caughtstealinghome)) as SB_percent_away
from game g join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id;

select ((r2dh.stolenbasehome/(r2dh.stolenbasehome+ r2dh.caughtstealinghome))-(r2da.stolenbasehome/(r2da.stolenbasehome+ 
r2da.caughtstealinghome))) as SB_percent_diff,
((r2dh.stolenbasehome/(r2dh.stolenbasehome+ r2dh.caughtstealinghome))/(r2da.stolenbasehome/(r2da.stolenbasehome+ 
r2da.caughtstealinghome))) -1.0 as SB_percent_ratio
from game g join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id;

# Combining all the stats
drop table features; 

create table features as(
select g.home_team_id,g.away_team_id, g.game_id,
((ifnull(r2dh.Hit,0)/nullif (r2dh.atBat,0)) /ifnull(r2da.Hit,0)/nullif (r2da.atBat,0)) -1.0 as BA_Ratio, ((ifnull(r2dh.Hit,0)/nullif(r2dh.atBat,0))- 
ifnull(r2da.Hit,0)/nullif (r2da.atBat,0)) as BA_diff,
(r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )/(r2dh.atBat)-(r2da.Single+2*r2da.`Double` +3*r2da.Triple
+4*r2da.Home_Run)/(r2da.atBat) as Slug_diff,((r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )/(r2dh.atBat))/((
r2da.Single+2*r2da.`Double` +3*r2da.Triple+4*r2da.Home_Run)/(r2da.atBat)) -1.0 as Slug_ratio,
((r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )-(r2da.Single+2*r2da.`Double` +3*r2da.Triple+4*r2da.Home_Run)) as total_bases_diff,
((r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )/(r2da.Single+2*r2da.`Double` +3*r2da.Triple+4*r2da.Home_Run))- 1.0 as total_bases_ratio,
((r2dh.`Double` +r2dh.Triple + r2dh.Home_Run )-(r2da.`Double` +r2da.Triple+r2da.Home_Run)) as extra_bases_lh_diff,
((r2dh.`Double` +r2dh.Triple + r2dh.Home_Run )/(r2da.`Double` +r2da.Triple+r2da.Home_Run)) - 1.0 as extra_bases_lh_ratio,
(((r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )+ r2dh.Hit_By_Pitch+ r2dh.stolenBaseHome + r2dh.walk)/((r2dh.atBat- r2dh.Hit)
+ r2dh.caughtStealingHome+ r2dh.Grounded_Into_Dp)) as Total_Average,
((r2dh.Hit+ r2dh.walk+ r2dh.Hit_By_Pitch)-(r2da.Hit+ r2da.walk+ r2da.Hit_By_Pitch)) times_on_bases_diff,
((r2dh.Hit+ r2dh.walk+ r2dh.Hit_By_Pitch)/(r2da.Hit+ r2da.walk+ r2da.Hit_By_Pitch))- 1.0 as times_on_bases_ratio,
(((r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch)/(r2dh.atBat+ r2dh.Walk+ r2dh.Hit_By_Pitch+ 
r2dh.Sac_Fly))- ((r2da.Hit+ r2da.Walk + r2da.Hit_By_Pitch)/(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly))) as on_base_diff,
(((r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch)/(r2dh.atBat+ r2dh.Walk+ r2dh.Hit_By_Pitch+ r2dh.Sac_Fly))/ 
((r2da.Hit+ r2da.Walk + r2da.Hit_By_Pitch)/(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly))) -1.0 as on_base_ratio,
((((r2dh.atBat*(r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch))+ ((r2dh.Single + 2*r2dh.`Double` 
+3*r2dh.Triple + 4*r2dh.Home_Run )*(r2dh.atBat+ r2dh.Walk+ r2dh.Hit_By_Pitch+ r2dh.Sac_Fly)))/(r2dh.atBat * (r2dh.atBat+ r2dh.Walk+ 
r2dh.Hit_By_Pitch+ r2dh.Sac_Fly)))-(((r2da.atBat*(r2da.Hit+ r2da.Walk + r2da.Hit_By_Pitch))+ ((r2da.Single + 2*r2da.`Double` 
+3*r2da.Triple + 4*r2dh.Home_Run )*(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly)))/(r2da.atBat * (r2da.atBat+ r2da.Walk+ 
r2da.Hit_By_Pitch+ r2da.Sac_Fly)))) as on_base_slug_diff,
((((r2dh.atBat*(r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch))+ ((r2dh.Single + 2*r2dh.`Double` 
+3*r2dh.Triple + 4*r2dh.Home_Run )*(r2dh.atBat+ r2dh.Walk+ r2dh.Hit_By_Pitch+ r2dh.Sac_Fly)))/(r2dh.atBat * (r2dh.atBat+ r2dh.Walk+ 
r2dh.Hit_By_Pitch+ r2dh.Sac_Fly)))/(((r2da.atBat*(r2da.Hit+ r2da.Walk + r2da.Hit_By_Pitch))+ ((r2da.Single + 2*r2da.`Double` 
+3*r2da.Triple + 4*r2dh.Home_Run )*(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly)))/(r2da.atBat * (r2da.atBat+ r2da.Walk+ 
r2da.Hit_By_Pitch+ r2da.Sac_Fly))))-1.0 as on_base_slug_ratio,
((ifnull(r2dh.atBat,0))/nullif(r2dh.Home_Run,0))- (ifnull(r2da.atBat,0)/nullif (r2da.Home_Run,0)) as at_bats_perHR_diff,
(((ifnull(r2dh.atBat,0))/nullif(r2dh.Home_Run,0))/(ifnull(r2da.atBat,0)/nullif (r2da.Home_Run,0)))- 1.0 as at_bats_perHR_ratio,
((r2dh.Home_Run/r2dh.Hit)-(r2da.Home_Run/r2da.Hit)) as hr_h_diff,
((r2dh.Home_Run/r2dh.Hit)/(r2da.Home_Run/r2da.Hit)) -1.0 as hr_h_ratio,
((r2dh.plateAppearance/r2dh.Strikeout)-(r2da.plateAppearance/r2da.Strikeout)) as PA_per_SO_diff,
((r2dh.plateAppearance/r2dh.Strikeout)/(r2da.plateAppearance/r2da.Strikeout)) -1.0 as PA_per_SO_ratio,
((r2dh.walk/r2dh.Strikeout)-(r2da.walk/r2da.Strikeout)) as walk_SO_diff,
((r2dh.walk/r2dh.Strikeout)/(r2da.walk/r2da.Strikeout)) as walk_SO_ratio,
(ifnull(r2dh.stolenbasehome,0)/nullif ((r2dh.stolenbasehome+ r2dh.caughtstealinghome),0))-(ifnull(r2da.stolenbasehome,0)/nullif ((r2da.stolenbasehome+ 
r2da.caughtstealinghome),0)) as SB_percent_diff,
(ifnull(r2dh.stolenbasehome,0)/nullif ((r2dh.stolenbasehome+ r2dh.caughtstealinghome),0))-(ifnull(r2da.stolenbasehome,0)/nullif ((r2da.stolenbasehome+ 
r2da.caughtstealinghome),0)) -1.0 as SB_percent_ratio,
(((1.8*((r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch)/(r2dh.atBat+ r2dh.Walk+ 
r2dh.Hit_By_Pitch+ r2dh.Sac_Fly))+(r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )/(r2dh.atBat))/4)-((1.8*((r2da.Hit+ 
r2da.Walk + r2da.Hit_By_Pitch)/(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly))+(r2da.Single + 2*r2da.`Double` +3*r2da.Triple + 
4*r2da.Home_Run )/(r2da.atBat))/4)) as gpa_diff,
(((1.8*((r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch)/(r2dh.atBat+ r2dh.Walk+ 
r2dh.Hit_By_Pitch+ r2dh.Sac_Fly))+(r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )/(r2dh.atBat))/4)/((1.8*((r2da.Hit+ 
r2da.Walk + r2da.Hit_By_Pitch)/(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly))+(r2da.Single + 2*r2da.`Double` +3*r2da.Triple + 
4*r2da.Home_Run )/(r2da.atBat))/4)) -1.0 as gpa_ratio,
case when b.away_runs < b.home_runs then 1
	   when b.away_runs > b.home_runs then 0 
	   else 0 end as Home_team_wins
from game g join RollingAvgTable r2da on r2da.team_id = g.away_team_id and g.game_id = r2da.game_id
join RollingAvgTable r2dh on r2dh.team_id = g.home_team_id and g.game_id = r2dh.game_id
join boxscore b on b.game_id = g.game_id;

select * from features; 



-- (r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run ) as Total_Bases_home, (r2da.Single+2*r2da.`Double` +3*r2da.Triple+4*r2da.Home_Run)as Total_Bases_away,
-- (r2dh.`Double` +r2dh.Triple + r2dh.Home_Run ) as Extra_Bases_home, 
-- (r2da.`Double` +r2da.Triple+r2da.Home_Run)as Extra_Bases_away,
-- (((r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )+ r2dh.Hit_By_Pitch+ r2dh.stolenBaseHome + r2dh.walk)/(
-- (r2dh.atBat- r2dh.Hit)+ r2dh.caughtStealingHome+ r2dh.Grounded_Into_Dp)) as Total_Average,
-- (r2dh.Hit+ r2dh.walk+ r2dh.Hit_By_Pitch) as Times_On_Bases_home,
-- (r2da.Hit+ r2da.walk+ r2da.Hit_By_Pitch) as Times_On_Bases_away,
-- ((r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch)/(r2dh.atBat+ r2dh.Walk+ r2dh.Hit_By_Pitch+ r2dh.Sac_Fly)) as on_base_percentage_home,
-- ((r2da.Hit+ r2da.Walk + r2da.Hit_By_Pitch)/(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly))as on_base_percentage_away,
-- (((r2dh.atBat*(r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch))+ ((r2dh.Single + 2*r2dh.`Double` 
-- +3*r2dh.Triple + 4*r2dh.Home_Run )*(r2dh.atBat+ r2dh.Walk+ r2dh.Hit_By_Pitch+ r2dh.Sac_Fly)))/(r2dh.atBat * (r2dh.atBat+ r2dh.Walk+ 
-- r2dh.Hit_By_Pitch+ r2dh.Sac_Fly)))as on_base_plus_slugging_home, 
-- (((r2da.atBat*(r2da.Hit+ r2da.Walk + r2da.Hit_By_Pitch))+ ((r2da.Single + 2*r2da.`Double` 
-- +3*r2da.Triple + 4*r2dh.Home_Run )*(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly)))/(r2da.atBat * (r2da.atBat+ r2da.Walk+ 
-- r2da.Hit_By_Pitch+ r2da.Sac_Fly)))as on_base_plus_slugging_away,
-- (ifnull(r2dh.stolenbasehome,0)/nullif((r2dh.stolenbasehome+ r2dh.caughtstealinghome),0)) as SB_percent_home ,
-- (ifnull(r2da.stolenbasehome,0)/nullif((r2da.stolenbasehome+ r2da.caughtstealinghome),0)) as SB_percent_away,
-- (ifnull(r2dh.atBat,0)/nullif(r2dh.Home_Run,0)) as at_bats_perHR_home,(ifnull(r2da.atBat,0)/nullif(r2da.Home_Run,0)) as at_bats_perHR_away,
-- (r2dh.plateAppearance/r2dh.Strikeout) as PA_per_SO_home,(r2da.plateAppearance/r2da.Strikeout) as PA_per_SO_away,
-- (r2dh.walk/r2dh.Strikeout) as walk_SO_home,(r2da.walk/r2da.Strikeout) as walk_SO_away,
-- (r2dh.Home_Run/r2dh.Hit) as hr_h_home,(r2da.Home_Run/r2da.Hit) as hr_h_away,
-- ((1.8*((r2dh.Hit+ r2dh.Walk + r2dh.Hit_By_Pitch)/(r2dh.atBat+ r2dh.Walk+ 
-- r2dh.Hit_By_Pitch+ r2dh.Sac_Fly))+(r2dh.Single + 2*r2dh.`Double` +3*r2dh.Triple + 4*r2dh.Home_Run )/(r2dh.atBat))/4) as gpa_home,
-- ((1.8*((r2da.Hit+ r2da.Walk + r2da.Hit_By_Pitch)/(r2da.atBat+ r2da.Walk+ r2da.Hit_By_Pitch+ r2da.Sac_Fly))+(r2da.Single + 
-- 2*r2da.`Double` +3*r2da.Triple + 4*r2da.Home_Run )/(r2da.atBat))/4) as gpa_away,


