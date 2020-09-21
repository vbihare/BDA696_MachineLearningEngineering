USE baseball;


#Calculating the Historic batting average
create or replace table HistBattingAvg(batter int, batting_average float) ENGINE=MyISAM
	SELECT batter,(sum(Hit)/sum(atBat)) as batting_average 
	from batter_counts 
	group by batter;

#Viewing the results
select * 
	from histbattingavg;

#Calculating the Annual Batting Average
create or replace table AnnualBatAvg(batter int, batting_average float) ENGINE=MyISAM
	Select b.batter, (SUM(Hit)/SUM(atBat)) as batting_average, YEAR(local_date)
	from batter_counts b 
	join game g 
	on b.game_id = g.game_id 
	group by b.batter,YEAR(local_date)
	order by b.batter;

#Viewing the results
select * 
	from Annualbatavg;


#Rolling average, game id chosen - 1
#lets go back 100 days to calculate batting averages of players involved in that game

create or replace table RollingBattingAvg(batter int, batting_average float, game_id int) ENGINE=MyISAM 
	Select b.batter, (SUM(Hit)/SUM(atBat)) as batting_average, g.game_id  
	from batter_counts b 
	join game g 
	on b.game_id = g.game_id 
	where b.batter in 
					(select batter from battersInGame big where game_id = '1') 
					and cast (local_date as date) 
					between 
					(select date_add((select cast(local_date as DATE) from game where game_id ='1'), 
					interval -100 DAY)) 
	and (select cast(local_date as DATE) from game where game_id ='1')
	group by batter;

#Viewing the results
select * 
	from rollingbattingavg;
