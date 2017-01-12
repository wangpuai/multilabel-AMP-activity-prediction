function Average_Precision = Metric_AveragePrecision(Outputs,test_target)
%Computing the average precision
%Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

    [num_class,num_instance] = size(Outputs);
    temp_Outputs = [];
    temp_test_target = [];
    for i = 1:num_instance
        temp = test_target(:,i);   %第i个测试样本的标记向量，1-(-1)向量
        if((sum(temp)~=num_class)&(sum(temp)~=-num_class))  %不全为1，或者不全为-1
            temp_Outputs = [temp_Outputs, Outputs(:,i)];
            temp_test_target = [temp_test_target, temp];
        end
    end
    Outputs = temp_Outputs;    %把类标记全为1和全为-1的测试样本去掉
    test_target = temp_test_target;     
    [num_class,num_instance] = size(Outputs);
    
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
    
    aveprec = 0;
    for i = 1:num_instance
        temp = Outputs(:,i);
        [~,index] = sort(temp);  %按照第i个样本具有各个标签的概率从小到大 排列所有标签并储存在index中
        indicator = zeros(1,num_class);
        for m = 1:Label_size(i)
            [~,loc] = ismember(Label{i,1}(m),index);  %第i个测试样本的标签集中第m个标签对应index中的位置
            indicator(1,loc) = 1;   %indicator为0-1向量，如[0 1 1 0]，实际上就是将第i个样本的对各标签的概率从小到大排列，并将具有某标签对应的概率置1，其它置0
        end
        summary = 0;
        for m = 1:Label_size(i)
            [~,loc] = ismember(Label{i,1}(m),index);
            summary = summary + sum(indicator(loc:num_class))/(num_class-loc+1);  %累加第i个样本具有的每个标签所对应的精度，如[0 1 1 0]，summary = 1/2 + 2/3
        end
%         ap_binary(i) = summary/Label_size(i);
        aveprec = aveprec + summary/Label_size(i);   %第i个样本的平均精度
    end
    Average_Precision = aveprec/num_instance;   %所有样本的平均精度
    