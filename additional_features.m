 datamat=load('data/a01/p1/s01.txt');
 [m,n]=size(datamat);
 additional_data=[];
 for addn_data_row=1:m
    addn_data_column_vec=[];
    for i=1:3:n
        value=(datamat(addn_data_row,i)^2)+(datamat(addn_data_row,i+1).^2)+(datamat(addn_data_row,i+2).^2);
        addn_data_column_vec=[addn_data_column_vec value];
    end
    additional_data=[additional_data;addn_data_column_vec];
 end
           