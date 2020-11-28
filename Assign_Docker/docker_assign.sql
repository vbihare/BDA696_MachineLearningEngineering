#########################################################################
							# BDA696 Assignment: 5
							# Rolling Avg using Docker
#########################################################################

USE baseball;

#Calculating the roling average Rolling average

create table if not exists batter as
	(select hit,atbat,batter,b.game_id,Date(local_date) as local_dt
	 from batter_counts b 
	 join game g 
	 on g.game_id = b.game_id 
	 ORDER BY game_id ,local_dt);
		 
	
create table if not exists rolling_avg as
	(select sum(ifnull(ba.Hit,0))/nullif(sum(ifnull(ba.atbat,0)),0) as Rolling_average, ba1.local_dt, ba1.batter, ba1.game_id, count(*) as cnt
	from batter ba1 join batter ba on ba.batter = ba1.batter and ba.local_dt between date_add(ba1.local_dt, interval -100 day) 
	and date_sub(ba1.local_dt, interval 1 day) 
	where ba1.game_id = 12560 
	group by ba1.game_id ,ba1.batter, ba1.local_dt);

#Converting it into text file
SELECT * from rolling_avg
INTO OUTFILE '/Assign_Docker/rolling_avg_vb.txt';
