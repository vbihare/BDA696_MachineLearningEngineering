#########################################################################
							# BDA696 Assignment: 2	
#########################################################################

USE baseball;


#Calculating the Historic batting average

create or replace table HistoricAvg(batter int, batting_average float) ENGINE=MyISAM
	SELECT batter,(sum(Hit)/sum(atBat)) as batting_average 
	from batter_counts 
	group by batter;

#Viewing the results
select * 
	from HistoricAvg;

#Calculating the Annual Batting Average

create or replace table AnnualBatAvg(batter int, batting_average float) ENGINE=MyISAM
	Select b.batter, (SUM(Hit)/SUM(atBat)) as batting_average, YEAR(local_date) as game_yr
	from batter_counts b 
	join game g 
	on b.game_id = g.game_id 
	group by b.batter, game_yr 
	order by b.batter;

#Viewing the results
select * 
	from Annualbatavg;


#Calculating the roling average Rolling average

create or replace table batter as
	(select hit,atbat,batter,b.game_id,Date(local_date) as local_dt
	 from batter_counts b 
	 join game g 
	 on g.game_id = b.game_id 
	 ORDER BY game_id ,local_dt);


create or replace table rolling_avg  as
	(select sum(ifnull(ba.Hit,0))/nullif(sum(ifnull(ba.atbat,0)),0) as Rolling_average, ba1.local_dt, ba1.batter, ba1.game_id, count(*) as cnt
	from batter ba1 join batter ba on ba.batter = ba1.batter and ba.local_dt > date_sub(ba1.local_dt, interval 100 day) 
	and ba.local_dt < ba1.local_dt
	#where ba1.game_id = 10000 #(remove comment to limit values to a particular game)
	group by ba1.game_id ,ba1.batter, ba1.local_dt);
	

#Viewing the results
select * 
	from rolling_avg;
