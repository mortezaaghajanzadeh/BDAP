* Annual accounting data for North American countries;
proc sql;
	create table funda as
	select gvkey, datadate, curcd, at, lt, seq, ib, 'na' as source
	from comp.funda
	where indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C';
quit;
* Annual accounting data for countries outside of North American;
proc sql;
	create table g_funda as
	select gvkey, datadate, curcd, at, lt, seq, ib, 'global' as source
	from comp.g_funda
	where indfmt in ('INDL', 'FS') and datafmt='HIST_STD' and popsrc='I' and consol='C';
quit;
* Combine NA and non-NA data;
data acc1;
	set funda g_funda;
run;
* Create a unique data set;
proc sort data=acc2 nodupkey; by gvkey datadate; run;

* Change data to USD;
%include "~/Global Data/project_macros.sas";
%compustat_fx(out=fx);

proc sql;
	create table acc3 as 
	select a.*, b.fx
	from acc2 as a left join fx as b
	on a.datadate=b.date and a.curcd=b.curcdd;
quit;

data acc4;
	set acc3;
	array var at lt seq ib;
	do over var;
		var = var*fx;
	end;
	curcd = 'USD';
	drop fx;
run;

* Improving coverage;
data acc5;
	set acc4;
	shareholders_equity = coalesce(seq, at-lt);
run;

* Characteristic;
data acc6;
	set acc5;
	return_on_assets = ib / at;
run;

* Public availability;
data acc7;
	set acc6;
	public_date = intnx('month', datadate, 4,'e'); format public_date YYMMDDN8.;;
run;