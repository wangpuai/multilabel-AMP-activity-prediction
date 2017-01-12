function RankingLoss = Metric_RankingLoss(Outputs, test_target)
%Computing the Ranking loss
%Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

    [num_class, num_instance] = size(Outputs);
    temp_Outputs = [];
    temp_test_target = [];
    for i = 1:num_instance
        temp = test_target(:,i);   %第i个测试样本的标记向量，1-(-1)向量
        if((sum(temp)~=num_class)&(sum(temp)~=-num_class))  %不全为1，或者不全为-1
            temp_Outputs = [temp_Outputs,Outputs(:,i)];
            temp_test_target = [temp_test_target,temp];
        end
    end
    Outputs = temp_Outputs;    %把类标记全为1和全为-1的测试样本去掉
    test_target = temp_test_target;
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
    
    rankloss = 0;
    for i = 1:num_instance
        temp = 0;
        for m = 1:Label_size(i)
            for n = 1:(num_class-Label_size(i))
                if(Outputs(Label{i,1}(m),i)<=Outputs(not_Label{i,1}(n),i))  %第i个测试样本具有第m个标签的概率<=不具有第n个标签的概率
                    temp = temp+1;  %对每个测试样本，累计上述情况发生的次数
                end
            end
        end
%         rl_binary(i) = temp/(m*n);
        rankloss = rankloss + temp/(m*n);  %累加每个测试样本的排序错误率，m：具有的标签数，n：不具有的标签数
    end
    RankingLoss = rankloss/num_instance;  %平均排序错误率
    