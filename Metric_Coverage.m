function Coverage = Metric_Coverage(Outputs, test_target)
%Computing the coverage
%Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

       [num_class, num_instance] = size(Outputs);
    
       Label = cell(num_instance,1);      %每个测试样本具有的标签集合
       not_Label = cell(num_instance,1);  %每个测试样本不具有的标签集合
       Label_size = zeros(1,num_instance);  %每个测试样本具有的标签数
       for i = 1:num_instance
           temp = test_target(:,i);
           Label_size(1,i) = sum(temp==ones(num_class,1));
           for j = 1:num_class
               if(temp(j)==1)
                   Label{i,1} = [Label{i,1},j];
               else
                   not_Label{i,1} = [not_Label{i,1},j];
               end
           end
       end

       cover = 0;
       for i = 1:num_instance
           temp = Outputs(:,i);
           [~,index] = sort(temp);  %按照第i个样本具有各个标签的概率从小到大 排列所有标签并储存在index中
           temp_min = num_class+1;
           for m = 1:Label_size(i)
               [~,loc] = ismember(Label{i,1}(m),index);  %第i个测试样本的标签集中第m个标签对应index中的位置
               if(loc<temp_min)
                   temp_min = loc;  %第i个测试样本的标签集所对应index中的最小位置，也就是找出该样本全部具有标签所需要的深度
               end
           end
           cover = cover+(num_class-temp_min+1);   %累加每个样本的搜索深度
       end
       Coverage = (cover/num_instance)-1;    %平均深度
       